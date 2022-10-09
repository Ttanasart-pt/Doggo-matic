const VIDEO = document.getElementById('webcam');
const CANVAS = document.getElementById('overlay');
const RESULT_P = document.getElementById('result_p');
const RESULT_I = document.getElementById('result_i');

let canvasCtx = CANVAS.getContext('2d');
let stream = null;
let streaming = false;

const videoHeight = 480;
const videoWidth = 640;
const FPS = 30;

const INPUT_WIDTH = 224;
const INPUT_HEIGHT = 224;
const CLASS_NAMES = [
    "Affenpinscher",
    "Afghan_hound",
    "Airedale terrier",
    "Akita",
    "Alaskan malamute",
    "American eskimo dog",
    "American foxhound",
    "American staffordshire terrier",
    "American water spaniel",
    "Anatolian shepherd dog",
    "Australian cattle dog",
    "Australian shepherd",
    "Australian terrier",
    "Basenji",
    "Basset hound",
    "Beagle",
    "Bearded collie",
    "Beauceron",
    "Bedlington terrier",
    "Belgian malinois",
    "Belgian sheepdog",
    "Belgian tervuren",
    "Bernese mountain dog",
    "Bichon frise",
    "Black and tan coonhound",
    "Black russian terrier",
    "Bloodhound",
    "Bluetick coonhound",
    "Border collie",
    "Border terrier",
    "Borzoi",
    "Boston terrier",
    "Bouvier des flandres",
    "Boxer",
    "Boykin spaniel",
    "Briard",
    "Brittany",
    "Brussels griffon",
    "Bull terrier",
    "Bulldog",
    "Bullmastiff",
    "Cairn terrier",
    "Canaan dog",
    "Cane corso",
    "Cardigan welsh corgi",
    "Cavalier king charles spaniel",
    "Chesapeake bay retriever",
    "Chihuahua",
    "Chinese crested",
    "Chinese shar-pei",
    "Chow chow",
    "Clumber spaniel",
    "Cocker spaniel",
    "Collie",
    "Curly-coated retriever",
    "Dachshund",
    "Dalmatian",
    "Dandie dinmont terrier",
    "Doberman pinscher",
    "Dogue de bordeaux",
    "English cocker spaniel",
    "English setter",
    "English springer spaniel",
    "English toy spaniel",
    "Entlebucher mountain dog",
    "Field spaniel",
    "Finnish spitz",
    "Flat-coated retriever",
    "French bulldog",
    "German pinscher",
    "German shepherd dog",
    "German shorthaired pointer",
    "German wirehaired pointer",
    "Giant schnauzer",
    "Glen of imaal terrier",
    "Golden retriever",
    "Gordon setter",
    "Great dane",
    "Great pyrenees",
    "Greater swiss mountain dog",
    "Greyhound",
    "Havanese",
    "Ibizan hound",
    "Icelandic sheepdog",
    "Irish red and white setter",
    "Irish setter",
    "Irish terrier",
    "Irish water spaniel",
    "Irish wolfhound",
    "Italian greyhound",
    "Japanese chin",
    "Keeshond",
    "Kerry blue terrier",
    "Komondor",
    "Kuvasz",
    "Labrador retriever",
    "Lakeland terrier",
    "Leonberger",
    "Lhasa apso",
    "Lowchen",
    "Maltese",
    "Manchester terrier",
    "Mastiff",
    "Miniature schnauzer",
    "Neapolitan mastiff",
    "Newfoundland",
    "Norfolk terrier",
    "Norwegian buhund",
    "Norwegian elkhound",
    "Norwegian lundehund",
    "Norwich terrier",
    "Nova scotia duck tolling retriever",
    "Old english sheepdog",
    "Otterhound",
    "Papillon",
    "Parson russell terrier",
    "Pekingese",
    "Pembroke welsh corgi",
    "Petit basset griffon vendeen",
    "Pharaoh hound",
    "Plott",
    "Pointer",
    "Pomeranian",
    "Poodle",
    "Portuguese water dog",
    "Saint bernard",
    "Silky terrier",
    "Smooth fox terrier",
    "Tibetan mastiff",
    "Welsh springer spaniel",
    "Wirehaired pointing griffon",
    "Xoloitzcuintli",
    "Yorkshire terrier",
];


CANVAS.addEventListener('click', enableCam);

function hasGetUserMedia() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

function enableCam() {
    if (streaming) return;
    navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false
        })
        .then(function (s) {
            stream = s;
            VIDEO.srcObject = s;
            VIDEO.play();
        })
        .catch(function (err) {
            console.log("An error occured! " + err);
        });

        VIDEO.addEventListener("canplay", function (ev) {
        if (!streaming) {
            VIDEO.videoWidth = videoWidth;
            VIDEO.videoHeight = videoHeight;
            CANVAS.width = videoWidth;
            CANVAS.height = videoHeight;
            streaming = true;
        }
        startVideoProcessing();
    }, false);
}

let faceClassifier = null;
let eyeClassifier = null;

let src = null;
let predicting = false;

let canvasInput = null;
let canvasInputCtx = null;

let canvasBuffer = null;
let canvasBufferCtx = null;

function startVideoProcessing() {
    if (!streaming) {
        console.warn("Please startup your webcam");
        return;
    }
    stopVideoProcessing();
    canvasInput = document.createElement('canvas');
    canvasInput.width = videoWidth;
    canvasInput.height = videoHeight;
    canvasInputCtx = canvasInput.getContext('2d');

    canvasBuffer = document.createElement('canvas');
    canvasBuffer.width = videoWidth;
    canvasBuffer.height = videoHeight;
    canvasBufferCtx = canvasBuffer.getContext('2d');

    srcMat = new cv.Mat(videoHeight, videoWidth, cv.CV_8UC4);
    grayMat = new cv.Mat(videoHeight, videoWidth, cv.CV_8UC1);

    faceClassifier = new cv.CascadeClassifier();
    faceClassifier.load('haarcascade_frontalface_alt.xml');

    requestAnimationFrame(processVideo);
}

function processVideo() {
    canvasInputCtx.drawImage(VIDEO, 0, 0, videoWidth, videoHeight);
    let imageData = canvasInputCtx.getImageData(0, 0, videoWidth, videoHeight);
    srcMat.data.set(imageData.data);
    cv.cvtColor(srcMat, grayMat, cv.COLOR_RGBA2GRAY);
    let faces = [];
    let size;
    let faceVect = new cv.RectVector();
    let faceMat = new cv.Mat();
    cv.pyrDown(grayMat, faceMat);
    cv.pyrDown(faceMat, faceMat);
    size = faceMat.size();
    faceClassifier.detectMultiScale(faceMat, faceVect);
    const pad = 16;
    for (let i = 0; i < faceVect.size(); i++) {
        let face = faceVect.get(i);
        face.x -= pad;
        face.y -= pad;
        face.width += pad * 2;
        face.height += pad * 2;
        faces.push(new cv.Rect(face.x, face.y, face.width, face.height));
        
        if(!predicting)
            predict(face);
    }
    faceMat.delete();
    faceVect.delete();
    canvasCtx.drawImage(canvasInput, 0, 0, videoWidth, videoHeight);
    drawResults(canvasCtx, faces, 'red', size);
    requestAnimationFrame(processVideo);
}

function drawResults(ctx, results, color, size) {
    for (let i = 0; i < results.length; ++i) {
        let rect = results[i];
        let xRatio = videoWidth / size.width;
        let yRatio = videoHeight / size.height;
        ctx.lineWidth = 2;
        ctx.strokeStyle = color;
        ctx.strokeRect(rect.x * xRatio, rect.y * yRatio, rect.width * xRatio, rect.height * yRatio);
    }
}

function stopVideoProcessing() {
    if (src != null && !src.isDeleted()) src.delete();
}

function stopCamera() {
    if (!streaming) return;
    stopVideoProcessing();
    canvasCtx.clearRect(0, 0, videoWidth, videoHeight);
    VIDEO.pause();
    VIDEO.srcObject = null;
    stream.getVideoTracks()[0].stop();
    streaming = false;
}

/////////////////////////////////////////////////////
async function predict(frame) {
    predicting = false;
    tf.tidy(function () {
        let videoFrameAsTensor = tf.browser.fromPixels(VIDEO).div(255);
        let boxes = [[frame.y / videoHeight, frame.x / videoWidth, (frame.y + frame.height) / videoHeight, (frame.x + frame.width) / videoWidth]];
        let resizedTensorFrame = tf.image.cropAndResize(videoFrameAsTensor.expandDims(), boxes, [0], [INPUT_HEIGHT, INPUT_WIDTH]);

        let imageFeatures = mobilenet.predict(resizedTensorFrame);
        let prediction = model.predict(imageFeatures).squeeze();
        let highestIndex = prediction.argMax().arraySync();
        RESULT_P.innerHTML = CLASS_NAMES[highestIndex]
        RESULT_I.src = `dogs/${highestIndex}.png`
    });
    predicting = false;
}

let mobilenet = undefined;
let videoPlaying = false;
let trainingDataInputs = [];
let trainingDataOutputs = [];
let examplesCount = [];
async function loadMobileNetFeatureModel() {
    const URL =
        'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1';

    mobilenet = await tf.loadGraphModel(URL, {
        fromTFHub: true
    });

    tf.tidy(function () {
        let answer = mobilenet.predict(tf.zeros([1, INPUT_HEIGHT, INPUT_WIDTH, 3]));
        console.log(answer.shape);
    });
}
loadMobileNetFeatureModel();

let model = tf.sequential();
model.add(tf.layers.dense({inputShape: [1024], units: 128, activation: 'relu'}));
model.add(tf.layers.dense({units: CLASS_NAMES.length, activation: 'softmax'}));
//model = tf.loadLayersModel('modelWeight');