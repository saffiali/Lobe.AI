using System;
using System.IO;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using lobe.ImageSharp;
using lobe;


namespace Lobe.AI.Helpers.Console
{
    class Program
    {
        static void Main(string[] args)
        {



            //var signatureFilePath = args[0];
            //var modelFile = args[1];
            //var modelFormat = args[2];
            //var imageToClassify = args[3];

            System.Console.WriteLine("signatureFilePath:");
            var signatureFilePath = System.Console.ReadLine();

            System.Console.WriteLine("modelFile:");
            var modelFile = System.Console.ReadLine();
            
            System.Console.WriteLine("modelFormat:");
            var modelFormat = System.Console.ReadLine();

            System.Console.WriteLine("imageToClassify:");
            var imageToClassify = System.Console.ReadLine();

            ImageClassifier.Register("onnx", () => new OnnxImageClassifier());
            using var classifier = ImageClassifier.CreateFromSignatureFile(
                new FileInfo(signatureFilePath),
                modelFile,
                modelFormat);

            var results = classifier.Classify(Image
                .Load(imageToClassify).CloneAs<Rgb24>());
            System.Console.WriteLine("This image lable is" +results.Classification.Label+" with confidence score:"+ results.Classification.Confidence);
        }
    }
}
