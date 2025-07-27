using LLama;
using LLama.Native;

namespace ModelFreeSpeculation;

/// <summary>
/// Draft provider for spell checking - uses original text as drafts
/// </summary>
public class TextSpellCheckDraftProvider(string originalText, LLamaWeights weights) : IDraftProvider
{
    private readonly LLamaToken[] _originalTokens = weights.Tokenize(originalText, false, false, System.Text.Encoding.UTF8);

    // This is the token search window, not the max edit distance (e.g., it's not Levenshtein distance).
    private const int AlignmentSearchRadius = 20;

    // State to track the last high-confidence alignment position (the "anchor")
    private int _lastGoodMatchOriginalPosition;
    private int _lastGoodMatchProcessedLength;
    private const int MinAnchorMatchLength = 5; // Increased for more stability

    public Task<IReadOnlyList<LLamaToken>> RequestDraftsAsync(
        IConversationContext context,
        int maxDrafts,
        CancellationToken cancellationToken = default)
    {
        var processedTokens = context.ProcessedTokens;

        // Find the best alignment between processed tokens and original tokens
        int originalPosition = FindBestAlignment(processedTokens);

        // Return the next tokens from the original sequence
        var remainingTokens = _originalTokens.Skip(originalPosition).Take(maxDrafts).ToList();
        return Task.FromResult<IReadOnlyList<LLamaToken>>(remainingTokens);
    }

    private int FindBestAlignment(IReadOnlyList<LLamaToken> processedTokens)
    {
        if (processedTokens.Count == 0)
        {
            _lastGoodMatchOriginalPosition = 0;
            _lastGoodMatchProcessedLength = 0;
            return 0;
        }

        var (position, foundAlignment) = FindBestTokenAlignment(processedTokens);

        if (!foundAlignment)
        {
            position = TryFallbackAlignment(processedTokens);
        }

        return position;
    }

    private int TryFallbackAlignment(IReadOnlyList<LLamaToken> processedTokens)
    {
        // Fallback 1: Use the anchor. This is the most reliable fallback.
        if (_lastGoodMatchProcessedLength > 0 && processedTokens.Count > _lastGoodMatchProcessedLength)
        {
            int tokensGeneratedSinceAnchor = processedTokens.Count - _lastGoodMatchProcessedLength;
            int estimatedPosition = _lastGoodMatchOriginalPosition + tokensGeneratedSinceAnchor;
            return Math.Min(estimatedPosition, _originalTokens.Length);
        }

        // Fallback 2: Final resort, align by count.
        return Math.Min(processedTokens.Count, _originalTokens.Length);
    }

    private (int position, bool foundAlignment) FindBestTokenAlignment(IReadOnlyList<LLamaToken> processedTokens)
    {
        // *** Use a suffix of processed tokens for local alignment ***
        int lookBackCount = Math.Min(processedTokens.Count, AlignmentSearchRadius * 2);
        var processedSuffix = processedTokens.Skip(processedTokens.Count - lookBackCount).ToList();

        // Estimate where the suffix *should* end in the original text
        int expectedEndPosition = _lastGoodMatchOriginalPosition + (processedTokens.Count - _lastGoodMatchProcessedLength);

        // Define a search window in the original text around the expected position
        int searchStart = Math.Max(0, expectedEndPosition - lookBackCount - AlignmentSearchRadius);
        int searchEnd = Math.Min(_originalTokens.Length, expectedEndPosition + AlignmentSearchRadius);

        var originalSearchWindow = _originalTokens
            .Skip(searchStart)
            .Take(searchEnd - searchStart)
            .ToList();

        if (originalSearchWindow.Count == 0)
        {
            return (_originalTokens.Length, false);
        }

        // Find the best alignment of the SUFFIX within the WINDOW
        var (bestScore, endPosInWindow) = FindBestSubsequenceAlignment(processedSuffix, originalSearchWindow);

        // Calculate the absolute position in the full _originalTokens array
        int bestPosition = searchStart + endPosInWindow;

        // Reject the alignment if the error rate is too high for the suffix
        double errorRate = lookBackCount > 0 ? (double)bestScore / lookBackCount : 0;
        if (errorRate > 0.6)
        {
            return (bestPosition, false); // Alignment is too poor, trigger fallback.
        }

        // ANCHOR LOGIC: Update anchor if we have a very strong recent match
        if (bestScore <= 2 && lookBackCount >= MinAnchorMatchLength)
        {
            _lastGoodMatchOriginalPosition = bestPosition;
            _lastGoodMatchProcessedLength = processedTokens.Count;
        }

        return (bestPosition, true);
    }

    private static (int cost, int endPosition) FindBestSubsequenceAlignment(IReadOnlyList<LLamaToken> sequenceToFind, IReadOnlyList<LLamaToken> sequenceToSearchIn)
    {
        int n = sequenceToFind.Count;
        int m = sequenceToSearchIn.Count;
        if (n == 0) return (0, 0);

        int[,] dp = new int[n + 1, m + 1];

        for (int i = 0; i <= n; i++) dp[i, 0] = i;
        for (int j = 0; j <= m; j++) dp[0, j] = 0;

        for (int i = 1; i <= n; i++)
        {
            for (int j = 1; j <= m; j++)
            {
                int cost = (sequenceToFind[i - 1] == sequenceToSearchIn[j - 1]) ? 0 : 1;

                int substitute = dp[i - 1, j - 1] + cost;
                int delete = dp[i - 1, j] + 1;
                int insert = dp[i, j - 1] + 1;

                dp[i, j] = Math.Min(substitute, Math.Min(delete, insert));
            }
        }

        int minCost = int.MaxValue;
        int endPosition = 0;
        for (int j = 0; j <= m; j++)
        {
            if (dp[n, j] <= minCost)
            {
                minCost = dp[n, j];
                endPosition = j;
            }
        }

        return (minCost, endPosition);
    }
}