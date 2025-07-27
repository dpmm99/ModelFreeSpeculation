using LLama;
using LLama.Batched;
using LLama.Native;

namespace ModelFreeSpeculation;

/// <summary>
/// Implementation of conversation context
/// </summary>
internal class ConversationContext(Conversation conversation, IReadOnlyList<LLamaToken> processedTokens, int position, LLamaContext llamaContext) : IConversationContext
{
    public Conversation Conversation { get; } = conversation;
    public IReadOnlyList<LLamaToken> ProcessedTokens { get; } = processedTokens;
    public int Position { get; } = position;
    public LLamaContext LlamaContext { get; } = llamaContext;

    public ConversationContext WithNewTokens(IReadOnlyList<LLamaToken> newTokens, int newPosition)
    {
        var allTokens = ProcessedTokens.Concat(newTokens).ToList();
        return new ConversationContext(Conversation, allTokens, newPosition, LlamaContext);
    }
}
