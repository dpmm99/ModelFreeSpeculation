using LLama;
using LLama.Batched;
using LLama.Common;
using LLama.Native;
using System.Diagnostics;
using System.Text;

namespace ModelFreeSpeculation;

/// <summary>
/// Core speculative decoding executor using BatchedExecutor
/// </summary>
public class SpeculativeDecodingExecutor(
    LLamaWeights model,
    ModelParams parameters,
    IDraftProvider draftProvider,
    SpeculativeDecodingConfig? config = null) : IDisposable
{
    private readonly BatchedExecutor _executor = new(model, parameters);
    private readonly SpeculativeDecodingConfig _config = config ?? new SpeculativeDecodingConfig();
    private readonly IDraftProvider _draftProvider = draftProvider;
    private bool _disposed;

    /// <summary>
    /// Execute speculative decoding with the given prompt
    /// </summary>
    public async Task<SpeculativeDecodingResult> ExecuteAsync(
        string prompt,
        CancellationToken cancellationToken = default)
    {
        var conversation = _executor.Create();
        //var startTokens = _executor.Context.Tokenize(prompt);
        // Due to LlamaSharp's design, we cannot prompt the conversation twice in a row without Infer().
        var template = new LLamaTemplate(_executor.Model);
        template.Add("user", prompt);
        template.AddAssistant = true;
        var templateBytes = template.Apply(); //TODO: One of these days, we have to find a better way than converting to a string and back to bytes again. The template application function shouldn't return a string!
        var startTokens = _executor.Context.Tokenize(Encoding.UTF8.GetString(templateBytes), true, true);

        var finalTokens = new List<LLamaToken>();
        var acceptedTokens = new List<LLamaToken>();
        var rejectedTokenCount = 0;
        var context = new ConversationContext(conversation, finalTokens, 0, _executor.Context);

        try
        {
            while (finalTokens.Count < _config.MaxTotalTokens && !cancellationToken.IsCancellationRequested && (finalTokens.Count == 0 || !finalTokens[^1].IsEndOfGeneration(model.Vocab)))
            {
                // Get draft tokens from provider
                var draftTokens = _config.MaxDraftTokens == 0
                    ? new List<LLamaToken>()
                    : await _draftProvider.RequestDraftsAsync(context, _config.MaxDraftTokens, cancellationToken);

                if (draftTokens.Count == 0)
                {
                    //Console.WriteLine("No draft tokens returned.");
                    //We still have to keep inferring until we reach the end, using token.IsEndOfGeneration(model.Vocab).
                    conversation.Prompt(startTokens ?? [model.Vocab.EOS.Value], true);

                    // Infer as long as needed
                    while (conversation.RequiresInference)
                    {
                        var result = await _executor.Infer(cancellationToken);
                        if (result != DecodeResult.Ok)
                            throw new InvalidOperationException($"Initial inference failed with result: {result}");
                    }

                    var finalSample = conversation.Sample();
                    var finalLogitsArray = LLamaTokenDataArray.Create(finalSample);
                    finalLogitsArray.Softmax();
                    var lastToken = finalLogitsArray.Data.Span[0].ID;
                    finalTokens.Add(lastToken);
                    startTokens = [lastToken];

                    continue;
                }

                //Write out the last 5 tokens (decoded) plus draft tokens
                //Console.WriteLine($"Last 5 Tokens: {string.Join("|", finalTokens.TakeLast(5).Select(t => _executor.Model.Vocab.LLamaTokenToString(t, false)))}");
                //Console.WriteLine($" Draft Tokens: {string.Join("|", draftTokens.Select(t => _executor.Model.Vocab.LLamaTokenToString(t, false)))}");

                // Evaluate drafts using batched executor
                var evaluationResult = await EvaluateDraftsAsync(conversation, startTokens != null ? [.. startTokens, .. draftTokens] : [.. draftTokens], draftTokens.Count, cancellationToken);
                startTokens = evaluationResult.NextPromptPrefix;

                // Process results
                finalTokens.AddRange(evaluationResult.Final);
                acceptedTokens.AddRange(evaluationResult.Accepted);
                rejectedTokenCount += evaluationResult.RejectedCount;

                // Update context for next iteration
                context = context.WithNewTokens(evaluationResult.Final, context.Position + evaluationResult.Final.Count);
            }

            // Decode final result
            var decoder = new StreamingTokenDecoder(_executor.Context);
            decoder.AddRange(finalTokens);
            var decodedText = decoder.Read();

            return new SpeculativeDecodingResult
            {
                FinalTokens = finalTokens.AsReadOnly(),
                AcceptedTokens = acceptedTokens.AsReadOnly(),
                RejectedTokenCount = rejectedTokenCount,
                IsComplete = acceptedTokens.Count >= _config.MaxTotalTokens,
                DecodedText = decodedText
            };
        }
        finally
        {
            conversation.Dispose();
        }
    }

    private async Task<DraftEvaluationResult> EvaluateDraftsAsync(
        Conversation conversation,
        List<LLamaToken> promptTokens,
        int draftTokenCount,
        CancellationToken cancellationToken)
    {
        // Prompt for ALL those tokens at once, because conversations have a strict prompt->infer->sample->prompt state loop.
        conversation.Prompt(promptTokens, true);

        // Infer as long as needed
        while (conversation.RequiresInference)
        {
            var result = await _executor.Infer(cancellationToken);
            if (result != DecodeResult.Ok)
                throw new InvalidOperationException($"Initial inference failed with result: {result}");
        }

        // Sample at each draft token position
        var final = new List<LLamaToken>();
        var accepted = new List<LLamaToken>();
        var rejectedCount = 0;
        for (var x = draftTokenCount; x > 0; x--)
        {
            var draftToken = promptTokens[^x];
            var sample = conversation.Sample(x);
            var logitsArray = LLamaTokenDataArray.Create(sample);
            logitsArray.Softmax();

            // Find the probability of our draft token
            var tokenProbability = 0f;
            var chosenToken = logitsArray.Data.Span[0];
            foreach (var item in logitsArray.Data.Span)
            {
                if (item.ID == draftToken)
                {
                    tokenProbability = item.Probability;
                    chosenToken = item;
                    break;
                }
                if (_config.AcceptanceThreshold == 1) break; // If we only accept exact matches, we can stop after evaluating just the first token
            }

            var isAccepted = (chosenToken.ID == draftToken && _config.AcceptanceThreshold == 1) || tokenProbability >= _config.AcceptanceThreshold;

            var eval = new TokenEvaluation(draftToken, tokenProbability, isAccepted, promptTokens.Count - x, chosenToken.ID, chosenToken.Probability);

            if (eval.IsAccepted)
            {
                accepted.Add(eval.Token);
                final.Add(eval.Token);
            }
            else
            {
                rejectedCount += x;
                // Use the top token from the logits as a fallback--just because the draft token was rejected doesn't mean we should waste the inference time.
                final.Add(eval.TopToken);

                conversation.Rewind(x); // Rewind to just after the last accepted draft token
                // Replace the first wrong draft token with the correct inferred+sampled one, except we CANNOT prompt here because we want to prompt again immediately!
                // Stop at first rejection
                return new DraftEvaluationResult(final, accepted, rejectedCount, [eval.TopToken]);
            }
        }

        // One more sample after the last accepted token--or we could change it to prompt one less token instead for maybe slightly better efficiency.
        var finalSample = conversation.Sample();
        var finalLogitsArray = LLamaTokenDataArray.Create(finalSample);
        finalLogitsArray.Softmax();
        var lastToken = finalLogitsArray.Data.Span[0].ID;
        final.Add(lastToken);
        return new DraftEvaluationResult(final, accepted, rejectedCount, [lastToken]);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _executor?.Dispose();
            _disposed = true;
            GC.SuppressFinalize(this);
        }
    }
}
