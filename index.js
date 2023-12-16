const express = require('express');
const cors = require('cors');
const multer = require('multer');
require('@tensorflow/tfjs-node');
const faceapi = require('face-api.js');
const canvas = require('canvas'); 
const { Canvas, Image, ImageData, createCanvas, loadImage } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 7857;

app.use(cors());
app.use(express.json());
app.use(express.static('images'));

// Configure multer to handle file uploads
const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

// Load face-api.js models
faceapi.nets.ssdMobilenetv1.loadFromDisk('./models');
faceapi.nets.faceLandmark68Net.loadFromDisk('./models');
faceapi.nets.faceRecognitionNet.loadFromDisk('./models');

// Endpoint for face recognition
app.post('/recognize', upload.single('image'), async (req, res) => {
  try {
    fs.writeFileSync(`./upload/${req.file.originalname}`, req.file.buffer)
    // console.log('///&&////: ',faceapi.);
    const referenceImage = await loadImage('./images/IMG_20231124_144003.jpg')
    // const queryImage = await loadImage(QUERY_IMAGE)


    // Now you can use the canvas as an image
    const results = await faceapi.detectAllFaces(referenceImage)
      .withFaceLandmarks()
      .withFaceDescriptors()
    console.log('//////: ',results);
    if (!results.length) {
        return
    }
    res.json({ results });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Internal Server Error' });
  }
});

app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
