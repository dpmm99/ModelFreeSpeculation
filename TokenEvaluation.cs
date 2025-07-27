using LLama.Native;

namespace ModelFreeSpeculation;

/// <summary>
/// Result of evaluating a single token
/// </summary>
internal record TokenEvaluation(
    LLamaToken Token,
    float Probability,
    bool IsAccepted,
    int Position,
    LLamaToken TopToken,
    float TopTokenProbability);
