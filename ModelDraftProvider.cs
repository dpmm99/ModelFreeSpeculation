using LLama;
using LLama.Batched;
using LLama.Common;
using LLama.Native;

namespace ModelFreeSpeculation;

/// <summary>
/// Draft provider using a smaller language model
/// </summary>
public class ModelDraftProvider(LLamaWeights draftModel, ModelParams parameters) : IDraftProvider
{
    private readonly BatchedExecutor _draftExecutor = new(draftModel, parameters);

    public async Task<IReadOnlyList<LLamaToken>> RequestDraftsAsync(
        IConversationContext context,
        int maxDrafts,
        CancellationToken cancellationToken = default)
    {
        var draftConversation = _draftExecutor.Create();

        try
        {
            // Prime the draft model with the same context
            draftConversation.Prompt(context.ProcessedTokens.ToList());

            var drafts = new List<LLamaToken>();

            for (int i = 0; i < maxDrafts; i++)
            {
                await _draftExecutor.Infer(cancellationToken);

                var sample = draftConversation.Sample();
                var logitsArray = LLamaTokenDataArray.Create(sample);

                if (logitsArray.Data.Length > 0)
                {
                    var token = logitsArray.Data.Span[0].ID;
                    drafts.Add(token);
                    draftConversation.Prompt(token);
                }
                else
                {
                    break;
                }
            }

            return drafts;
        }
        finally
        {
            draftConversation.Dispose();
        }
    }

    public void Dispose()
    {
        _draftExecutor?.Dispose();
    }
}