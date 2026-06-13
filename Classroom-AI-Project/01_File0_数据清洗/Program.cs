using System;
using System.IO;
using OpenCvSharp;

namespace EmotionDataPrep
{
    class Program
    {
        // ================== 参数配置 ==================
        private const string SourceRoot = @"D:\LINSHENG数据库";
        private const string TargetRoot = @"D:\LINSHENG_Clean_Images";
        // 人脸模型 XML 文件的路径，请根据实际存放位置修改
        private const string FaceModelPath = @"D:\LINSHENG数据库\haarcascade_frontalface_default.xml";

        private const int TargetFps = 5;       // 我们期望稀释后的帧率
        private const int OriginalFps = 60;   // 原始视频帧率
        private const int TargetSize = 224;    // CNN 预训练模型标准输入的尺寸 (224x224)
        // ==============================================

        static void Main(string[] sender)
        {
            Console.WriteLine("🚀 开始执行数据清洗与帧率稀释程序...");

            if (!File.Exists(FaceModelPath))
            {
                Console.WriteLine($"❌ 错误：未找到人脸模型文件，请确保路径正确: {FaceModelPath}");
                return;
            }

            // 1. 初始化人脸检测器
            using var faceCascade = new CascadeClassifier(FaceModelPath);

            // 2. 计算抽样步长（60帧 / 5帧 = 每隔 12 帧抽取一帧）
            int frameStep = OriginalFps / TargetFps;

            // 3. 遍历 D 盘根目录下的子文件夹（1专注、2困惑等）
            var subDirectories = Directory.GetDirectories(SourceRoot);

            foreach (var subDir in subDirectories)
            {
                string dirName = Path.GetFileName(subDir);
                string targetSubDir = Path.Combine(TargetRoot, dirName);

                // 创建对应的目标清洗文件夹
                if (!Directory.Exists(targetSubDir))
                {
                    Directory.CreateDirectory(targetSubDir);
                }

                Console.WriteLine($"\n📁 正在处理分类文件夹: {dirName}");

                // 4. 读取当前文件夹下的所有视频文件
                var videoFiles = Directory.GetFiles(subDir, "*.*", SearchOption.TopDirectoryOnly);

                foreach (var videoPath in videoFiles)
                {
                    string ext = Path.GetExtension(videoPath).ToLower();
                    if (ext != ".mp4" && ext != ".avi" && ext != ".mov") continue; // 过滤非视频文件

                    string videoName = Path.GetFileNameWithoutExtension(videoPath);
                    Console.WriteLine($"  🎬 正在解析视频: {videoName}{ext}");

                    // 打开视频流
                    using var capture = new VideoCapture(videoPath);
                    if (!capture.IsOpened())
                    {
                        Console.WriteLine($"  ⚠️ 无法打开视频文件: {videoPath}");
                        continue;
                    }

                    using var frame = new Mat();
                    int frameCount = 0;
                    int savedCount = 0;

                    // 5. 逐帧读取视频
                    while (capture.Read(frame))
                    {
                        // 帧率稀释逻辑：只处理符合步长的帧（例如第 0, 12, 24 帧...）
                        if (frameCount % frameStep == 0 && !frame.Empty())
                        {
                            // 转换为灰度图（提高人脸检测的准确率和速度）
                            using var grayFrame = new Mat();
                            Cv2.CvtColor(frame, grayFrame, ColorConversionCodes.BGR2GRAY);

                            // 执行人脸检测
                            var faces = faceCascade.DetectMultiScale(
                                grayFrame,
                                scaleFactor: 1.1,
                                minNeighbors: 5,
                                minSize: new Size(80, 80) // 过滤掉太小的人脸噪点
                            );

                            // 如果检测到了人脸（通常课堂单人视频只有一张脸，取第一张即可）
                            if (faces.Length > 0)
                            {
                                var faceRect = faces[0];

                                // 空间裁剪：把人脸区域（ROI）从原图里抠出来
                                using var croppedFace = new Mat(frame, faceRect);

                                // 尺寸规范化：将抠出来的人脸缩放到 224x224
                                using var resizedFace = new Mat();
                                Cv2.Resize(croppedFace, resizedFace, new Size(TargetSize, TargetSize));

                                // 6. 保存清洗后的图片到目标文件夹
                                string imageName = $"{videoName}_f{frameCount}.jpg";
                                string savePath = Path.Combine(targetSubDir, imageName);
                                Cv2.ImWrite(savePath, resizedFace);

                                savedCount++;
                            }
                        }
                        frameCount++;
                    }
                    Console.WriteLine($"  ✅ 视频处理完成。共抽检 {frameCount} 帧，成功裁剪并保存纯净人脸图片: {savedCount} 张");
                }
            }

            Console.WriteLine("\n🎉 🎉 🎉 所有视频预处理清洗完毕！");
            Console.WriteLine($"请前往查看成果: {TargetRoot}");
            Console.ReadLine();
        }
    }
}