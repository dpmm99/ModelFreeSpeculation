using LLama.Native;

namespace ModelFreeSpeculation;

/// <summary>
/// Result of evaluating draft tokens
/// </summary>
internal record DraftEvaluationResult(
    IReadOnlyList<LLamaToken> Final,
    IReadOnlyList<LLamaToken> Accepted,
    int RejectedCount,
    LLamaToken[]? NextPromptPrefix);

