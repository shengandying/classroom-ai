using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Vision;

namespace EmotionModelTrainer
{
    // 1. 定义输入数据结构
    public class ImageData
    {
        public string? ImagePath { get; set; }
        public string? Label { get; set; }
    }

    // 2. 定义预测输出结构
    public class ModelOutput
    {
        public string? PredictedLabel { get; set; }
        public float[]? Score { get; set; } // 存储每个分类的概率百分比
    }

    class Program
    {
        private const string CleanImagesRoot = @"D:\LINSHENG_Clean_Images";
        private const string ModelSavePath = @"D:\EmotionModel.zip";

        static void Main(string[] args)
        {
            Console.WriteLine("🚀 开始启动 CNN 迁移学习模型训练程序...");

            if (!Directory.Exists(CleanImagesRoot))
            {
                Console.WriteLine($"❌ 错误：未找到清洗后的图片文件夹，请先运行 File 0！路径: {CleanImagesRoot}");
                return;
            }

            // 1. 初始化 ML.NET 上下文环境
            var mlContext = new MLContext(seed: 1);

            Console.WriteLine("📂 正在加载图片数据并自动标注标签...");
            // 2. 自动遍历文件夹，以子文件夹名字（1专注、2困惑等）作为标签（Label）
            var images = LoadImagesFromDirectory(CleanImagesRoot);

            if (!images.Any())
            {
                Console.WriteLine("⚠️ 文件夹内没有找到任何图片，请检查 File 0 是否成功生成图片。");
                return;
            }

            Console.WriteLine($"📊 成功加载数据！总计图片数量: {images.Count} 张");

            // 3. 将数据转换为 ML.NET 原生支持的 IDataView 格式，并随机打乱数据
            var shuffledData = mlContext.Data.ShuffleRows(mlContext.Data.LoadFromEnumerable(images));

            // 4. 划分数据集：80% 用于训练，20% 用于评估模型准不准
            var trainTestSplit = mlContext.Data.TrainTestSplit(shuffledData, testFraction: 0.2);
            var trainSet = trainTestSplit.TrainSet;
            var testSet = trainTestSplit.TestSet;

            Console.WriteLine("🏗️ 正在构建 CNN 迁移学习流水线 (基于 ResNet V2 架构)...");
            // 5. 构建数据处理与训练流水线
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelKey", inputColumnName: "Label") // 将文本标签转为计算机认识的键值
                .Append(mlContext.Transforms.LoadRawImageBytes(outputColumnName: "ImageBytes", imageFolder: null, inputColumnName: "ImagePath")) // 读取图片原生字节流
                .Append(mlContext.MulticlassClassification.Trainers.ImageClassification(new ImageClassificationTrainer.Options()
                {
                    FeatureColumnName = "ImageBytes",
                    LabelColumnName = "LabelKey",
                    Arch = ImageClassificationTrainer.Architecture.ResnetV250, // 使用经典的 ResNet V2 50层深度网络
                    Epoch = 100,          // 训练轮数，200张小图 50轮 极快
                    BatchSize = 32,       // 批处理大小
                    LearningRate = 0.005f, // 学习率
                    MetricsCallback = (metrics) => Console.WriteLine(metrics.ToString()), // 在控制台实时打印训练进度
                    //ValidationSet = testSet // 使用 20% 的验证集进行同步测试
                }))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: "PredictedLabel", inputColumnName: "PredictedLabel")); // 训练完把键值转回人类看得懂的文本标签

            Console.WriteLine("🏋️‍♂️ 模型开始训练（微调中），请稍候...");
            // 6. 开始训练模型
            var trainedModel = pipeline.Fit(trainSet);
            Console.WriteLine("✅ 训练完成！");

            Console.WriteLine("📊 正在对训练好的模型进行质量评估...");
            // 7. 评估模型准确率
            var predictions = trainedModel.Transform(testSet);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: "LabelKey", scoreColumnName: "Score");

            Console.WriteLine($"====== 模型评估报告 ======");
            Console.WriteLine($"★ 总体准确率 (Macro Accuracy): {metrics.MacroAccuracy:P2}");
            Console.WriteLine($"★ 微观准确率 (Micro Accuracy): {metrics.MicroAccuracy:P2}");
            Console.WriteLine($"==========================");

            Console.WriteLine($"💾 正在将学会认表情的模型导出到本地...");
            // 8. 保存训练好的模型文件
            mlContext.Model.Save(trainedModel, trainSet.Schema, ModelSavePath);

            Console.WriteLine($"🎉 模型导出成功！文件保存在: {ModelSavePath}");
            Console.WriteLine("💡 这个 EmotionModel.zip 就是你智能体未来要加载的『大脑核心』。");
            Console.ReadLine();
        }

        // 辅助方法：遍历 D 盘清洗后的文件夹
        private static List<ImageData> LoadImagesFromDirectory(string path)
        {
            var list = new List<ImageData>();
            var subDirs = Directory.GetDirectories(path);

            foreach (var dir in subDirs)
            {
                string labelName = Path.GetFileName(dir); // 文件夹名作为 Label
                var files = Directory.GetFiles(dir, "*.jpg");

                foreach (var file in files)
                {
                    list.Add(new ImageData
                    {
                        ImagePath = file,
                        Label = labelName
                    });
                }
            }
            return list;
        }
    }
}