using LLama;
using LLama.Batched;
using LLama.Native;

namespace ModelFreeSpeculation;

/// <summary>
/// Context information passed to draft providers
/// </summary>
public interface IConversationContext
{
    /// <summary>
    /// Current conversation state
    /// </summary>
    Conversation Conversation { get; }

    /// <summary>
    /// All tokens processed so far in this sequence
    /// </summary>
    IReadOnlyList<LLamaToken> ProcessedTokens { get; }

    /// <summary>
    /// Current position in the original sequence (for text-based drafts)
    /// </summary>
    int Position { get; }

    /// <summary>
    /// Access to the underlying context for tokenization, etc.
    /// </summary>
    LLamaContext LlamaContext { get; }
}
