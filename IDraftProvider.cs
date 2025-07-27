using LLama.Native;

namespace ModelFreeSpeculation;

/// <summary>
/// Interface for draft token providers. This allows different "draft models" including:
/// - Original text for spell checking
/// - Smaller language models
/// - Rule-based systems
/// - Human input
/// </summary>
public interface IDraftProvider
{
    /// <summary>
    /// Request draft tokens based on current context
    /// </summary>
    /// <param name="context">Current conversation context</param>
    /// <param name="maxDrafts">Maximum number of draft tokens to provide</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Sequence of draft tokens to evaluate</returns>
    Task<IReadOnlyList<LLamaToken>> RequestDraftsAsync(
        IConversationContext context,
        int maxDrafts,
        CancellationToken cancellationToken = default);
}
