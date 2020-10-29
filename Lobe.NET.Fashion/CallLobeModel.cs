using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Azure.WebJobs;
using Microsoft.Azure.WebJobs.Extensions.Http;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using lobe.ImageSharp;
using lobe;

namespace Lobe.AI.Helpers
{
    public static class CallLobeModel
    {
        [FunctionName("CallLobeModel")]
        public static async Task<IActionResult> Run(
            [HttpTrigger(AuthorizationLevel.Function, "get", "post", Route = null)] HttpRequest req,
            ILogger log, ExecutionContext context)
        {
            log.LogInformation("Lobe CallLobeModel function was triggered.");
            try
            {
                string imageToClassify = req.Query["imageToClassify"];
                string requestBody = await new StreamReader(req.Body).ReadToEndAsync();
                dynamic data = JsonConvert.DeserializeObject(requestBody);
                imageToClassify = imageToClassify ?? data?.imageToClassify;

                LobeResult results = CallModel(imageToClassify, context);
                log.LogInformation("Lobe CallLobeMode finished inference");

                return new OkObjectResult(results);
            }
            catch (Exception e)
            {
                log.LogError("Lobe CallLobeModel error: " + e.Message);
                return new BadRequestObjectResult(e.Message);
            }
            
        }

        private static string GetConfig(string AppSettings)
        {
            return Environment.GetEnvironmentVariable(AppSettings) ?? "";
        }

        private static LobeResult CallModel(string imageToClassify, ExecutionContext context)
        {
            try
            {
                imageToClassify= System.IO.Path.Combine(context.FunctionDirectory, "..\\testImages\\1565.jpg");

                var signatureFilePath = System.IO.Path.Combine(context.FunctionDirectory, "..\\signature.json"); ;
                var modelFile = System.IO.Path.Combine(context.FunctionDirectory, "..\\model.onnx");
                var modelFormat = GetConfig("modelFormat");

                ImageClassifier.Register("onnx", () => new OnnxImageClassifier());
                using var classifier = ImageClassifier.
                    CreateFromSignatureFile(
                    new FileInfo(signatureFilePath),
                    modelFile,
                    modelFormat);

                var results = classifier.Classify(Image
                    .Load(imageToClassify).CloneAs<Rgb24>());

                return new LobeResult { Confidence = results.Classification.Confidence, Label = results.Classification.Label };
            }
            catch (Exception e)
            {
                return new LobeResult
                {
                    Label = "unknown",
                    Confidence = 0
                };
            }
        }
    }

    public class LobeResult
    {
        public double Confidence { get; set; }
        public string Label { get; set; }
    }
}
