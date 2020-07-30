using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Runtime.InteropServices;

using Emgu.CV;
using Emgu.CV.Util;
using Emgu.CV.CvEnum;
using Emgu.Util.TypeEnum;
using Emgu.CV.Structure;

using RemovePaperTexture.Test;

namespace RemovePaperTexture
{
    public partial class Form1 : Form
    {

        Mat srcImage = new Mat();
        Mat FFTImage = new Mat();
        Mat newImg = new Mat();
        Image<Gray, byte> final;
        public Form1()
        {
            InitializeComponent();
        }

        private void FileItemClicked(object sender, ToolStripItemClickedEventArgs e)
        {
            
        }

        private void loadImageToolStripMenuItem_Click(object sender, EventArgs e)
        {
            // read image
            using (OpenFileDialog ofd = new OpenFileDialog())
            {
                ofd.Filter = "Image|*.jpg;*.bmp;*.png";

                if (ofd.ShowDialog() == DialogResult.OK)
                {
                    srcImage = new Mat(ofd.FileName,0);
                    tempImg.Image = srcImage.ToImage<Gray, byte>().ToBitmap();
                    
                }
                
            }
        }

        // DFT button
        private void button1_Click(object sender, EventArgs e)
        {
            // padd image because dft must be 2n width and hight
            int m = CvInvoke.GetOptimalDFTSize(srcImage.Rows);
            int n = CvInvoke.GetOptimalDFTSize(srcImage.Cols);
            Mat padded = new Mat();
            CvInvoke.CopyMakeBorder(srcImage, padded, 0, m - srcImage.Rows, 0, n - srcImage.Cols, BorderType.Constant);
            padded.ConvertTo(padded, DepthType.Cv32F); 
           
            Mat zeroMat = Mat.Zeros(padded.Rows, padded.Cols, DepthType.Cv32F, 1);
            
            VectorOfMat matVector = new VectorOfMat(); 
            matVector.Push(padded);
            matVector.Push(zeroMat); 
           // make a complex mat
            Mat complexI = new Mat(padded.Size, DepthType.Cv32F, 2);
            CvInvoke.Merge(matVector, complexI); 


            Mat fourier = new Mat(complexI.Size, DepthType.Cv32F, 2);
           // do dft
            CvInvoke.Dft(complexI, fourier, DxtType.Forward, complexI.Rows);


            // temp is to show result of dft
            Mat temp = Magnitude(fourier);
            
            temp = new Mat(temp, new Rectangle(0, 0, temp.Cols & -2, temp.Rows & -2));
            SwitchQuadrants(ref temp);
            CvInvoke.Normalize(temp, temp, 1.0, 0.0, NormType.MinMax, DepthType.Cv32F);
            CvInvoke.Imshow("Fourier Transform", temp);

            // FFTImage is to save dft infomation(complex) , only complex can be inverse
            FFTImage = fourier.Clone();


            // do with cross part
            // crop infomation of src image
            
            Mat Real = new Mat(fourier.Size, DepthType.Cv32F, 1);
            Mat Imaginary = new Mat(fourier.Size, DepthType.Cv32F, 1);

            VectorOfMat channels = new VectorOfMat();
            CvInvoke.Split(fourier, channels); 
            Real = channels.GetOutputArray().GetMat(0);
            Imaginary = channels.GetOutputArray().GetMat(1);

            
            SwitchQuadrants(ref Real);
            //CvInvoke.Normalize(Real, Real, 1.0, 0.0, NormType.MinMax, DepthType.Cv32F);
            SwitchQuadrants(ref Imaginary);
            //CvInvoke.Normalize(Imaginary, Imaginary, 1.0, 0.0, NormType.MinMax, DepthType.Cv32F);

            // Array data
            // convert to image instead of using Mat's data pointer
            Image<Gray, float> img_R = Real.ToImage<Gray, float>();
            Image<Gray, float> img_I = Real.ToImage<Gray, float>();

            Array tmpR = Real.GetData();
            // make a Real Image copy
            Array realCopy = Real.GetData();
            Image<Gray, float> copy_R = Real.ToImage<Gray, float>();
            for (int i = 0; i < img_R.Width; i++)
            {
                for (int j = 0; j < img_R.Height; j++)
                {
                    copy_R.Data[j, i, 0] = (float)realCopy.GetValue(j, i);
                }
            }
            CvInvoke.MedianBlur(copy_R, copy_R, 3);
            CvInvoke.Imshow("Copy Real Image", copy_R);

            Array tmpI = Imaginary.GetData();

            // some num
            int Center_w = img_R.Width/2;
            int Center_h = img_I.Height/2;
            int biaspixel = 8;

            for(int i=0;i <img_R.Width;i++)
            {
                for(int j = 0; j < img_R.Height; j++)
                {
                    if((i>= Center_w-biaspixel && i <=Center_w + biaspixel))
                    {
                        img_R.Data[j, i, 0] = (float)tmpR.GetValue(j, i);
                        img_I.Data[j, i, 0] = (float)tmpI.GetValue(j, i);
                    }
                    else if((j >= Center_h - biaspixel && j <= Center_h + biaspixel))
                    {
                        img_R.Data[j, i, 0] = (float)tmpR.GetValue(j, i);
                        img_I.Data[j, i, 0] = (float)tmpI.GetValue(j, i);
                        
                    }
                   else
                    {
                        img_R.Data[j, i, 0] = copy_R.Data[j, i, 0];
                        img_I.Data[j, i, 0] = 0;
                    }
                    
                }
            }

            CvInvoke.Imshow("R", img_R);
            //CvInvoke.Imshow("I", img_I);
            

            // Image back to Mat
            // make some merge
            Mat temp1 = new Mat(img_R.Size, DepthType.Cv32F, 1);
            Mat temp2 = new Mat(img_R.Size, DepthType.Cv32F, 1);
            temp1 = img_R.Mat;
            temp2 = img_I.Mat;
            VectorOfMat matVectorTemp = new VectorOfMat();
            matVectorTemp.Push(temp1);
            matVectorTemp.Push(temp2);
            Mat merge_mat = new Mat(img_R.Size, DepthType.Cv32F, 2);
            CvInvoke.Merge(matVectorTemp, merge_mat);

            FFTImage = merge_mat.Clone();
           
            //CvInvoke.Imshow("Fourier Transform R", merge_mat.ToBitmap());
            /////////////////////////////
            ///
            Mat magnitudeImage = Magnitude(fourier);
            magnitudeImage = new Mat(magnitudeImage, new Rectangle(0, 0, magnitudeImage.Cols & -2, magnitudeImage.Rows & -2));
            
            SwitchQuadrants(ref magnitudeImage);
           
            CvInvoke.Normalize(magnitudeImage, magnitudeImage, 1.0, 0.0, NormType.MinMax, DepthType.Cv32F);
            //CvInvoke.Imshow("Fourier Transform", magnitudeImage);
            
           
            CvInvoke.Normalize(magnitudeImage, magnitudeImage, 0, 255, NormType.MinMax, DepthType.Cv8U);
            pictureBox1.Image = magnitudeImage.ToBitmap();
            magnitudeImage.ToBitmap().Save("Temp.png");
           
            
        }

        private Mat Magnitude(Mat fftData)
        {
            
            Mat Real = new Mat(fftData.Size, DepthType.Cv32F, 1);
            
            Mat Imaginary = new Mat(fftData.Size, DepthType.Cv32F, 1);
            VectorOfMat channels = new VectorOfMat();
            CvInvoke.Split(fftData, channels); //将多通道mat分离成几个单通道mat
            Real = channels.GetOutputArray().GetMat(0);
            Imaginary = channels.GetOutputArray().GetMat(1);
            CvInvoke.Pow(Real, 2.0, Real);
            CvInvoke.Pow(Imaginary, 2.0, Imaginary);
            CvInvoke.Add(Real, Imaginary, Real);
            CvInvoke.Pow(Real, 0.5, Real);
            Mat onesMat = Mat.Ones(Real.Rows, Real.Cols, DepthType.Cv32F, 1);
            CvInvoke.Add(Real, onesMat, Real);
            CvInvoke.Log(Real, Real); //求自然对数
            return Real;
        }
        private Mat MagnitudeInverse(Mat fftData)
        {
           
            Mat Real = new Mat(fftData.Size, DepthType.Cv32F, 1);
            
            Mat Imaginary = new Mat(fftData.Size, DepthType.Cv32F, 1);
            VectorOfMat channels = new VectorOfMat();
            CvInvoke.Split(fftData, channels); 
            Real = channels.GetOutputArray().GetMat(0);
            Imaginary = channels.GetOutputArray().GetMat(1);

            
            CvInvoke.Pow(Real, 2.0, Real);
            CvInvoke.Pow(Imaginary, 2.0, Imaginary);
            CvInvoke.Add(Real, Imaginary, Real);
            CvInvoke.Pow(Real, 0.5, Real);
            Console.WriteLine(Real);

            return Real;
        }

        private void SwitchQuadrants(ref Mat mat)
        {
            int cx = mat.Cols / 2;
            int cy = mat.Rows / 2;
            Mat q0 = new Mat(mat, new Rectangle(0, 0, cx, cy)); 
            Mat q1 = new Mat(mat, new Rectangle(cx, 0, cx, cy)); 
            Mat q2 = new Mat(mat, new Rectangle(0, cy, cx, cy)); 
            Mat q3 = new Mat(mat, new Rectangle(cx, cy, cx, cy)); 
            Mat temp = new Mat(q0.Size, DepthType.Cv32F, 1);
            
            q0.CopyTo(temp);
            q3.CopyTo(q0);
            temp.CopyTo(q3);
            
            q1.CopyTo(temp);
            q2.CopyTo(q1);
            temp.CopyTo(q2);
        }


        

        // Inverse button 
        private void button2_Click(object sender, EventArgs e)
        {
            
            CvInvoke.Dft(FFTImage, FFTImage, DxtType.Inverse, FFTImage.Rows);
            Mat magnitudeImage = MagnitudeInverse(FFTImage);

            CvInvoke.Normalize(magnitudeImage, magnitudeImage, 1.0, 0.0, NormType.MinMax, DepthType.Cv32F);
            CvInvoke.Imshow("Fourier Transform Inverse", magnitudeImage);
            CvInvoke.Normalize(magnitudeImage, magnitudeImage, 0, 255, NormType.MinMax, DepthType.Cv8U);
            magnitudeImage.ToBitmap().Save("Temp.png");

        }

        // click on image right
        private void pictureBox1_Click(object sender, EventArgs e)
        {
           
            using (OpenFileDialog ofd = new OpenFileDialog())
            {
                ofd.Filter = "Image|*.jpg;*.bmp;*.png";

                if (ofd.ShowDialog() == DialogResult.OK)
                {
                    newImg = new Mat(ofd.FileName, 0);
                    pictureBox1.Image = newImg.ToImage<Gray, byte>().ToBitmap();
                }
                
            }
        }

        // PSNR
        private void button3_Click(object sender, EventArgs e)
        {
            
            label1.Text = Convert.ToString( CvInvoke.PSNR(srcImage, newImg));
        }
    }
}

