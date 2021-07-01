using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using TIS.Imaging;

namespace ImagingSource
{
    public partial class Form1 : Form
    {
        private int counter = 1;
        private ICImagingControl ic = new ICImagingControl();
        private VCDButtonProperty softwareTrigger = null;
        int StartImage;

        // The variable "TimerCurrentImage" is used for the index of the currently displayed image,
        // due auto repeat of the image sequence is running.
        int TimerCurrentImage;

        // "Images" will contain a copy of the image ring buffer, in order to get faster acces to the images. 
        TIS.Imaging.ImageBuffer[] Images;

        // The variable "FrameCount" is used to count the capture frames. If "FrameCount"
        // is greater than the image ring buffer size, the "btnCapture" will be enabled and
        // the user knows the image sequence in the ring buffer is now filled up completely.
        // FrameCount will be set to 0 in btnLiveVideo_Click after the live video has been started.
        // It is incremented in the ImageAvailable event handler.
        int FrameCount = 0;

        // "Seconds" determins the length of the image ring buffer in seconds. If the image sequence should 
        // contain more or less seconds, only the value of "Seconds" need to be changed here.
        int Seconds = 10;
        public Form1()
        {
            InitializeComponent();
            this.SoftwareTrigger();
        }

        public void SoftwareTrigger()
        {
            icImagingControl1.LiveCaptureContinuous = true; //Call ImageAvailable event for new images.
            icImagingControl1.LiveCaptureLastImage = false; // Do not save an image on live stop.
            // Add the ImageAvailable handler to the IC Imaging Control object.
            icImagingControl1.ImageAvailable += new EventHandler<ICImagingControl.ImageAvailableEventArgs>(ic_ImageAvailable);
            icImagingControl1.ShowDeviceSettingsDialog(); // Select a video capture device
            if (!icImagingControl1.DeviceValid)
                return;

            // Query the trigger mode property for enabling the trigger mode    
            VCDSwitchProperty TriggerMode = (VCDSwitchProperty)icImagingControl1.VCDPropertyItems.FindInterface(VCDIDs.VCDID_TriggerMode, VCDIDs.VCDElement_Value, VCDIDs.VCDInterface_Switch);
            if (TriggerMode == null)
                return;

            // If trigger mode is available, query the software trigger property
            softwareTrigger = (VCDButtonProperty)icImagingControl1.VCDPropertyItems.FindInterface(VCDIDs.VCDID_TriggerMode, VCDIDs.VCDElement_SoftwareTrigger, VCDIDs.VCDInterface_Button);
            if (softwareTrigger == null)
                return;

            TriggerMode.Switch = true; // Enable trigger mode,

            icImagingControl1.LiveStart(); // start the camera. No images are streamed, because trigger mode is enabled
        }

        // ImageAvailable event handler. The parameter "e" contains the imagebuffer with the
        // currently snapped image.
        // The image is saved as JPEG file.
        void ic_ImageAvailable(object sender, ICImagingControl.ImageAvailableEventArgs e)
        {
            //string filename = String.Format("dataset/{0}.jpg", counter);
            string filename = DateTime.Now.ToString("yyyy'-'MM'-'dd'_'HH'-'mm'-'ss"+counter);
            e.ImageBuffer.SaveAsJpeg($"dataset 2/{filename}.jpg", 70);
            counter++;
        }

        private void button1_MouseClick(object sender, MouseEventArgs e)
        {
            softwareTrigger.Push();
            //if (icImagingControl1.DeviceValid)
            //{
            //    if (icImagingControl1.LiveVideoRunning)
            //    {
            //        icImagingControl1.LiveStop();
            //    }
            //    else
            //    {
            //        icImagingControl1.LiveDisplay = true; //show live video
            //        timer1.Enabled = false;
            //        icImagingControl1.LiveStart();
            //        FrameCount = 0;
            //    }
            //}
        }

        private void button2_Click(object sender, EventArgs e)
        {
            icImagingControl1.LiveStop();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            icImagingControl1.LiveCaptureLastImage = false;
            // In order to capture all delivered frames the property "LiveCaptureContinuous" must be set 
            // to true.
            icImagingControl1.LiveCaptureContinuous = true;
            timer1.Enabled = false;

            // Since no video capture device has been selected now, all 
            // buttons except the "Device" button are disabled.


            // Resize the live display to the size of IC Imaging Control in order to display
            // the complete live video.
            icImagingControl1.LiveDisplayDefault = false;
            icImagingControl1.LiveDisplayHeight = icImagingControl1.Height;
            icImagingControl1.LiveDisplayWidth = icImagingControl1.Width;
        }

        private void timer1_Tick(object sender, EventArgs e)
        {
            DisplayTheImage(TimerCurrentImage);

            TimerCurrentImage = TimerCurrentImage + 1;
            if (TimerCurrentImage >= icImagingControl1.ImageRingBufferSize)
            {
                TimerCurrentImage = 0;
            }


        }


        /// <summary>
        /// DisplayTheImage
        /// 
        /// "DisplayTheImage()" displayed the image in the ring buffer specified by
        /// the parameter "index". The specified "index" will be mapped to the sequence
        /// of images in the ring buffer. This means "index" is added to "StartIndex". If
        /// "Index" plus "StartIndex" is greater than the ImageRingBufferSize, then the
        /// new index in the ring buffer must start at 0. 
        /// </summary>
        /// <param name="Index">The index of the image to be displayed.</param>
        private void DisplayTheImage(int Index)
        {
            int i;
            if (!icImagingControl1.LiveVideoRunning)
            {
                i = StartImage + Index;
                // Handling a possible overflow, in case the i is greater then the ring buffer size.
                if (i >= icImagingControl1.ImageRingBufferSize)
                {
                    i = i - icImagingControl1.ImageRingBufferSize;
                }
                icImagingControl1.DisplayImageBuffer(Images[i]);
            }
        }
    }
}
