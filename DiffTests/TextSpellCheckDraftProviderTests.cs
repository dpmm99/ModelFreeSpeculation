using LLama;
using LLama.Common;
using LLama.Native;
using Moq;
using System.Reflection;

namespace ModelFreeSpeculation.Tests;

[TestClass]
public class TextSpellCheckDraftProviderTests
{
#pragma warning disable CS8618
    private LLamaWeights _weights;
    private Mock<IConversationContext> _mockContext;
#pragma warning restore CS8618

    [TestInitialize]
    public void Setup()
    {
        var parameters = new ModelParams(@"C:\AI\Lite-Mistral-150M-v2-Instruct-Q6_K_L.gguf");
        _weights = LLamaWeights.LoadFromFile(parameters);
        _mockContext = new Mock<IConversationContext>();
    }

    [TestCleanup]
    public void Cleanup()
    {
        _weights.Dispose();
    }

    /// <summary>
    /// Helper method to simulate a token-by-token generation process, which is how the provider is used in reality.
    /// </summary>
    private async Task<(List<LLamaToken> draft, int finalPosition)> SimulateGeneration(TextSpellCheckDraftProvider provider, List<LLamaToken> processedTokens)
    {
        // Setup the mock context to return the current list of processed tokens
        _mockContext.Setup(c => c.ProcessedTokens).Returns(processedTokens);

        // Request a draft
        var draft = await provider.RequestDraftsAsync(_mockContext.Object, 5);

        // To verify the final position, we need to access the private method.
        var findAlignmentMethod = typeof(TextSpellCheckDraftProvider).GetMethod("FindBestAlignment", BindingFlags.NonPublic | BindingFlags.Instance);
        var finalPosition = (int)findAlignmentMethod!.Invoke(provider, new object[] { processedTokens })!;

        return (draft.ToList(), finalPosition);
    }

    [TestMethod]
    public async Task SimplePerfectMatch_StaysAligned()
    {
        var original = "This is a simple test.";
        var provider = new TextSpellCheckDraftProvider(original, _weights);
        var originalTokens = _weights.Tokenize(original, false, false, System.Text.Encoding.UTF8).ToList();

        // Simulate generating the first 4 tokens perfectly
        var processed = originalTokens.Take(4).ToList();
        var (draft, finalPos) = await SimulateGeneration(provider, processed);

        // Assert
        Assert.AreEqual(4, finalPos, "Position should be at the end of the processed tokens.");
        CollectionAssert.AreEqual(originalTokens.Skip(4).Take(5).ToList(), draft, "Draft should be the next tokens from the original text.");
    }

    [TestMethod]
    public async Task SingleSubstitution_RecoversAlignment()
    {
        var original = "This is a simple test.";
        var misspelled = "This is a smple test."; // "simple" -> "smple"
        var provider = new TextSpellCheckDraftProvider(original, _weights);

        var originalTokens = _weights.Tokenize(original, false, false, System.Text.Encoding.UTF8).ToList();
        var processedTokens = _weights.Tokenize(misspelled, false, false, System.Text.Encoding.UTF8).ToList();

        // The tokenization might differ in length; find the alignment point.
        string targetWord = "simple";
        int endOfWordIndex = original.IndexOf(targetWord) + targetWord.Length;
        string textUpToTarget = original.Substring(0, endOfWordIndex);

        // The expected position is the number of tokens that make up the text leading up to and including the target word.
        var expectedEndPosition = _weights.Tokenize(textUpToTarget, false, false, System.Text.Encoding.UTF8).Length + 1;
        var expectedDraft = originalTokens.Skip(expectedEndPosition).Take(5).ToList();

        var (draft, finalPos) = await SimulateGeneration(provider, processedTokens.Take(processedTokens.Count - 1).ToList()); // Simulate up to "smple"

        Assert.AreEqual(expectedEndPosition, finalPos, "Should align correctly after the substituted word.");
        CollectionAssert.AreEqual(expectedDraft, draft, "Draft should continue from the corrected position.");
    }

    [TestMethod]
    public async Task MultipleErrors_MaintainsAlignmentWithAnchors()
    {
        var original = "The quick brown fox jumps over the lazy dog. A second sentence for length.";
        var misspelled = "The quick brown fox jmps over a lazy dog. A secon sentence for length.";
        var provider = new TextSpellCheckDraftProvider(original, _weights);

        var originalTokens = _weights.Tokenize(original, false, false, System.Text.Encoding.UTF8).ToList();
        var processedTokens = _weights.Tokenize(misspelled, false, false, System.Text.Encoding.UTF8).ToList();

        var (draft, finalPos) = await SimulateGeneration(provider, processedTokens);

        // We expect the alignment to be at the end of the original sentence, despite errors.
        Assert.AreEqual(originalTokens.Count, finalPos, "Should successfully align to the end despite multiple errors.");
        Assert.AreEqual(0, draft.Count, "Draft should be empty as we've reached the end.");
    }

    [TestMethod]
    public async Task TotalMismatch_FallsBackToCount()
    {
        var original = "This is a test sentence.";
        var processedText = "A completely different phrase.";
        var provider = new TextSpellCheckDraftProvider(original, _weights);

        var processedTokens = _weights.Tokenize(processedText, false, false, System.Text.Encoding.UTF8).ToList();

        var (draft, finalPos) = await SimulateGeneration(provider, processedTokens);

        // With total mismatch, the fallback is to align by count.
        int expectedPosition = Math.Min(processedTokens.Count, _weights.Tokenize(original, false, false, System.Text.Encoding.UTF8).Count());

        Assert.AreEqual(expectedPosition, finalPos, "On total mismatch, should fall back to aligning by token count.");
    }

    [TestMethod]
    public async Task OffByOneError_RecoversAndAligns()
    {
        // Test case where "the" is inserted, a common error.
        var original = "jumps over lazy dog.";
        var processedText = "jumps over the lazy dog.";
        var provider = new TextSpellCheckDraftProvider(original, _weights);

        var originalTokens = _weights.Tokenize(original, false, false, System.Text.Encoding.UTF8).ToList();
        var processedTokens = _weights.Tokenize(processedText, false, false, System.Text.Encoding.UTF8).ToList();

        var (draft, finalPos) = await SimulateGeneration(provider, processedTokens);

        // It should align correctly to the end of the original text.
        Assert.AreEqual(originalTokens.Count, finalPos, "Should align to the end of the original string, ignoring the insertion.");
        Assert.AreEqual(0, draft.Count, "Draft should be empty at the end.");
    }

    [TestMethod]
    public async Task EmptyProcessed_StartsFromBeginning()
    {
        var original = "Start from the beginning.";
        var provider = new TextSpellCheckDraftProvider(original, _weights);
        var originalTokens = _weights.Tokenize(original, false, false, System.Text.Encoding.UTF8).ToList();

        var (draft, finalPos) = await SimulateGeneration(provider, new List<LLamaToken>());

        Assert.AreEqual(0, finalPos, "Position should be 0 for empty processed tokens.");
        CollectionAssert.AreEqual(originalTokens.Take(5).ToList(), draft, "Draft should be the first 5 tokens.");
    }
}