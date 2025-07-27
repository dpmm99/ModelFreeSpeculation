using LLama.Native;
using System.Collections.ObjectModel;

namespace ModelFreeSpeculation;

/// <summary>
/// Result of speculative decoding operation
/// </summary>
public class SpeculativeDecodingResult
{
    public IReadOnlyList<LLamaToken> AcceptedTokens { get; init; } = [];
    public int RejectedTokenCount { get; init; } = 0;
    public int AcceptanceRate => AcceptedTokens.Count * 100 / (AcceptedTokens.Count + RejectedTokenCount);
    public bool IsComplete { get; init; }
    public string DecodedText { get; init; } = string.Empty;
    public ReadOnlyCollection<LLamaToken> FinalTokens { get; internal set; }
}
