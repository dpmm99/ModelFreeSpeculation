namespace ModelFreeSpeculation;

/// <summary>
/// Configuration for speculative decoding
/// </summary>
public class SpeculativeDecodingConfig
{
    /// <summary>
    /// Maximum number of draft tokens to evaluate in parallel
    /// </summary>
    public int MaxDraftTokens { get; init; } = 8;

    /// <summary>
    /// Minimum probability threshold for accepting a draft token. Set to 1 to require it to be the top choice.
    /// </summary>
    public float AcceptanceThreshold { get; init; } = 1f;

    /// <summary>
    /// Maximum total tokens to generate/process
    /// </summary>
    public int MaxTotalTokens { get; init; } = 8192;
}
