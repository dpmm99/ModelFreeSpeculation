using LLama;
using LLama.Common;
using LLama.Native;
using Microsoft.Extensions.Configuration;
using System.Diagnostics;

namespace ModelFreeSpeculation;

internal static class Program
{
    static async Task Main(string[] args)
    {
        var configurationRoot = new ConfigurationBuilder()
            .AddJsonFile("appsettings.json", optional: true, reloadOnChange: true)
            .AddEnvironmentVariables()
            .Build();
        var configSection = configurationRoot.GetSection("SpeculativeDecoding");

        var parameters = new ModelParams(configSection["ModelPath"] ?? throw new Exception("No GGUF file specified."))
        {
            ContextSize = configSection.GetValue("ContextSize", configSection.GetValue<uint>("MaxTokens", 4096)),
            BatchSize = configSection.GetValue("BatchSize", (uint)4096),
            Threads = configSection.GetValue("Threads", Environment.ProcessorCount),
            GpuLayerCount = configSection.GetValue("GpuLayerCount", 999),
            //FlashAttention = true,
            TensorBufferOverrides = [.. configSection.GetValue("TensorBufferOverrides", string.Empty)!.Split(';', StringSplitOptions.RemoveEmptyEntries)
                .Select(p => p.Split('=', StringSplitOptions.RemoveEmptyEntries))
                .Select(p => new LLama.Abstractions.TensorBufferOverride(p[0], p[1]))],
            TensorSplits = new LLama.Abstractions.TensorSplitsCollection([..
                configSection.GetValue("TensorSplits", string.Empty)!.Split(',', StringSplitOptions.RemoveEmptyEntries).Select(float.Parse)]),
            TypeK = Enum.Parse<GGMLType>(configSection.GetValue("TypeK", nameof(GGMLType.GGML_TYPE_F16))!, true), //Note: Other than both Q8_0 or F16, these options generally just crash llama.cpp. Tried F16/Q8_0 and Q8_0/Q4_0, for example, and those both crash. So it pretty much has to be F16/F16 or Q8_0/Q8_0.
            TypeV = Enum.Parse<GGMLType>(configSection.GetValue("TypeV", nameof(GGMLType.GGML_TYPE_F16))!, true),
        };
        var model = await LLamaWeights.LoadFromFileAsync(parameters);

        var speculativeDecodingConfig = new SpeculativeDecodingConfig
        {
            MaxDraftTokens = configSection.GetValue("MaxDraftTokens", 8),
            AcceptanceThreshold = configSection.GetValue("AcceptanceThreshold", 1f),
            MaxTotalTokens = configSection.GetValue("MaxTotalTokens", 4096)
        };

        string command, originalText;
        if (configSection.GetValue("ParamsMethod", "") == nameof(SpellCheckTestParams))
        {
            (command, originalText) = SpellCheckTestParams();
        }
        else if (configSection.GetValue("ParamsMethod", "") == nameof(CodeEditingTestParams))
        {
            (command, originalText) = CodeEditingTestParams();
        }
        else
        {
            command = configSection.GetValue<string>("Command") ?? throw new Exception("No command specified.");
            originalText = configSection.GetValue<string>("OriginalText") ?? throw new Exception("No original text specified.");
        }

        var spellCheckProvider = new TextSpellCheckDraftProvider(originalText, model);
        var executor = new SpeculativeDecodingExecutor(model, parameters, spellCheckProvider, speculativeDecodingConfig);
        var stopwatch = Stopwatch.StartNew();
        var result = await executor.ExecuteAsync(command + "\n\n" + originalText);
        stopwatch.Stop();
        var durationSeconds = stopwatch.Elapsed.TotalSeconds;
        var tokensPerSecond = durationSeconds > 0 ? result.FinalTokens.Count / durationSeconds : 0;

        Console.WriteLine("\nFinal Decoded Text:");
        Console.WriteLine(result.DecodedText);
        Console.Write($"\nDuration: {durationSeconds:F2}s, Tokens: {result.FinalTokens.Count}, Tokens/sec: {tokensPerSecond:F2} ");
        if (speculativeDecodingConfig.MaxDraftTokens == 0)
        {
            Console.WriteLine("(baseline)");
        }
        else
        {
            Console.WriteLine($"({result.AcceptedTokens.Count} draft tokens accepted; {speculativeDecodingConfig.MaxDraftTokens} draft tokens per batch)");
            Console.WriteLine("Rejected Tokens: " + result.RejectedTokenCount);
        }
    }

    static (string, string) SpellCheckTestParams()
    {
        return ("Please repeat the following text but with all spelling and grammar errors corrected:",
            "The quick brown fox jmps over the lazy dog. While the sun was shinning brighly, a group of children played happilly in the park. Their laughter echoed througout the trees, and the birds sang beautifuly. However, not everone noticed the small, colorfull butterfly that flutered past. In the distnce, a man read a newspapper, completly unaware of the world around him. As the afternon turned to evening, the air grew cooler and the sky began to darken. Some of the children forgoten their jackets, but they were too busy to care. Eventualy, the park empted, leaving only the sound of the wind rustling through the leavs and the occassional chirp of a cricket.");
    }

    static (string, string) CodeEditingTestParams()
    {
        return ("Please refactor the following C# code to use dependency injection for the ILogger implementation. Only write the updated code; do not acknowledge:",
        """
        ```csharp
        using System;

        public interface ILogger
        {
            void Log(string message);
        }

        public class ConsoleLogger : ILogger
        {
            public void Log(string message)
            {
                Console.WriteLine($"[LOG] {message}");
            }
        }

        public class Calculator
        {
            private readonly ILogger _logger;

            public Calculator()
            {
                _logger = new ConsoleLogger();
            }

            public int Add(int a, int b)
            {
                _logger.Log($"Adding {a} and {b}");
                return a + b;
            }

            public int Subtract(int a, int b)
            {
                _logger.Log($"Subtracting {b} from {a}");
                return a - b;
            }

            public int Multiply(int a, int b)
            {
                _logger.Log($"Multiplying {a} and {b}");
                return a * b;
            }

            public int Divide(int a, int b)
            {
                if (b == 0)
                {
                    _logger.Log("Attempted division by zero.");
                    throw new DivideByZeroException();
                }
                _logger.Log($"Dividing {a} by {b}");
                return a / b;
            }
        }

        class Program
        {
            static void Main(string[] args)
            {
                var calc = new Calculator();
                Console.WriteLine(calc.Add(2, 3));
                Console.WriteLine(calc.Subtract(5, 2));
                Console.WriteLine(calc.Multiply(3, 4));
                Console.WriteLine(calc.Divide(10, 2));
            }
        }
        ```
        """);
    }
}
