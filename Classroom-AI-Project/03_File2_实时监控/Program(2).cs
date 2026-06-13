using Microsoft.ML;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;

namespace EmotionLiveInference
{
    // 1. 保持与 File 1 完全一致的数据结构
    public class ImageData
    {
        public string? ImagePath { get; set; }
        public string? Label { get; set; }
    }

    public class ModelOutput
    {
        public string? PredictedLabel { get; set; }
        public float[]? Score { get; set; }
    }

    class Program
    {
        // ================== 参数配置 ==================
        private const string ModelPath = @"D:\EmotionModel.zip";
        private const string FaceModelPath = @"D:\LINSHENG数据库\haarcascade_frontalface_default.xml";
        // 临时缓存图：用于把当前帧的人脸喂给模型（无缝适配 File 1 的 Schema）
        private const string TempLiveFacePath = @"D:\temp_live_face.jpg";
        // 最终日志输出路径
        private const string LogOutputPath = @"D:\LINSHENG_情感状态日志.csv";

        private const int TargetSize = 224; // 缩放到模型需要的 224x224
        // ==============================================

        static void Main(string[] args)
        {
            Console.WriteLine("🚀 正在启动林盛课堂实时情感智能体...");

            // 1. 检查必要文件
            if (!File.Exists(ModelPath) || !File.Exists(FaceModelPath))
            {
                Console.WriteLine("❌ 错误：缺少 EmotionModel.zip 或人脸 XML 模型，请检查 D 盘！");
                return;
            }

            // 2. 初始化 ML.NET 并加载大脑模型
            var mlContext = new MLContext();
            DataViewSchema modelSchema;
            var trainedModel = mlContext.Model.Load(ModelPath, out modelSchema);
            // 创建预测引擎
            var predictionEngine = mlContext.Model.CreatePredictionEngine<ImageData, ModelOutput>(trainedModel);
            Console.WriteLine("🧠 AI 情感模型加载成功！");

            // 3. 初始化 OpenCV 摄像头和人脸裁刀
            using var faceCascade = new CascadeClassifier(FaceModelPath);
            using var capture = new VideoCapture(0); // 0 代表电脑默认自带摄像头
            if (!capture.IsOpened())
            {
                Console.WriteLine("❌ 错误：无法打开摄像头，请检查设备连接。");
                return;
            }

            // 4. 初始化日志数据流与计时器
            List<string> emotionLogList = new List<string>();
            Stopwatch logTimer = new Stopwatch();
            logTimer.Start();

            Console.WriteLine("\n==============================================");
            Console.WriteLine("🖥️  实时监测已开启！弹出视频窗口后：");
            Console.WriteLine("👉 画面会实时显示当前学生的情感状态");
            Console.WriteLine("👉 程序每 10 秒会自动记录一次状态到日志");
            Console.WriteLine("👉 【重要】在画面窗口上按下『ESC』键，可安全结束并导出日志！");
            Console.WriteLine("==============================================\n");

            using var frame = new Mat();
            string currentPrediction = "未检测到人脸"; // 全局缓存当前状态供日志抓取

            // 5. 实时主循环
            while (capture.Read(frame) && !frame.Empty())
            {
                // 转灰度图用于人脸检测
                using var grayFrame = new Mat();
                Cv2.CvtColor(frame, grayFrame, ColorConversionCodes.BGR2GRAY);

                var faces = faceCascade.DetectMultiScale(grayFrame, 1.1, 5, minSize: new OpenCvSharp.Size(100, 100));

                if (faces.Length > 0)
                {
                    var faceRect = faces[0];

                    // 在原视频画面上画出绿色的正方形人脸框
                    Cv2.Rectangle(frame, faceRect, Scalar.Green, 2);

                    // 空间裁剪与尺寸规范化
                    using var croppedFace = new Mat(frame, faceRect);
                    using var resizedFace = new Mat();
                    Cv2.Resize(croppedFace, resizedFace, new OpenCvSharp.Size(TargetSize, TargetSize));

                    // 核心小技巧：保存到 D 盘临时文件，秒喂给预测引擎
                    Cv2.ImWrite(TempLiveFacePath, resizedFace);

                    // 让大模型进行实时推断
                    var inputData = new ImageData { ImagePath = TempLiveFacePath };
                    var result = predictionEngine.Predict(inputData);

                    currentPrediction = result.PredictedLabel ?? "未知";

                    // 将实时预测文字渲染到摄像头画面的左上角
                    Cv2.PutText(frame, $"Status: {currentPrediction}", new OpenCvSharp.Point(20, 50),
                                HersheyFonts.HersheySimplex, 1.0, Scalar.Red, 2);
                }
                else
                {
                    currentPrediction = "未检测到人脸";
                    Cv2.PutText(frame, "Status: No Face Detected", new OpenCvSharp.Point(20, 50),
                                HersheyFonts.HersheySimplex, 1.0, Scalar.Yellow, 2);
                }

                // 🌟 核心日志机制：每 10 秒记录一次情感数据
                if (logTimer.ElapsedMilliseconds >= 10000)
                {
                    string timeStamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
                    string logEntry = $"[{timeStamp}] ,{currentPrediction}";

                    emotionLogList.Add(logEntry); // 存入连续日志数组
                    Console.WriteLine($"📝 [日志捕获] {logEntry}"); // 控制台同步提示

                    logTimer.Restart(); // 重置 10 秒计时器
                }

                // 显示实时监控画面
                Cv2.ImShow("LINSHENG Classroom AI Monitor", frame);

                // 🌟 安全结束程序方式：检测用户是否在窗口按下 ESC 键（ASCII 码 27）
                int key = Cv2.WaitKey(1);
                if (key == 27)
                {
                    Console.WriteLine("\n🛑 检测到按下 ESC 键，正在安全关闭系统并生成报告...");
                    break; // 跳出循环
                }
            }

            // 6. 退出后的日志输出问题（完美落盘为 Excel 识别的 CSV）
            Console.WriteLine("💾 正在写入 Excel 表格日志到本地磁盘...");
            try
            {
                // 🔄 修改后：第一行直接定义 Excel 的列名
                List<string> finalReportOutput = new List<string>
    {
        "时间,学生情感状态" // A1单元格是“时间”，B1单元格是“学生情感状态”
    };
                finalReportOutput.AddRange(emotionLogList);

                // 🔄 修改后：使用带 BOM 的 UTF-8 编码，确保 Excel 打开时中文不乱码
                File.WriteAllLines(LogOutputPath, finalReportOutput, new System.Text.UTF8Encoding(true));

                Console.WriteLine($"🎉 导出成功！请直接用 Excel 双击打开文件: {LogOutputPath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ 写入日志文件失败: {ex.Message}");
            }

            // 7. 清理临时缓存文件
            if (File.Exists(TempLiveFacePath)) File.Delete(TempLiveFacePath);

            Console.WriteLine("\n👋 智能体系统已安全退出。谢谢使用！");
            Console.ReadLine();
        }
    }
}