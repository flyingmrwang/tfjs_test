import "@tensorflow/tfjs-backend-wasm";
// tfjsWasm.setWasmPath('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@latest/dist/tfjs-backend-wasm.wasm');


const webcamElement = document.getElementById('webcam');
// const tf = require('@tensorflow/tfjs');
// const tfnode = require('@tensorflow/tfjs-node');

let net;

async function loadModel() {
  await tf.setBackend('wasm')
  console.log(tf.getBackend())
  const model = await tf.loadLayersModel('http://localhost:8000/model.json'); //keras converted 
  
  // const model = await tf.loadGraphModel('http://localhost:8000/model.json'); //tf converted
  // tf.loadGraphModel("https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_050_224/classification/3/default/1", { fromTFHub: true })
  console.log('model file loaded');
  // const model = await tf.loadLayersModel('file:///Users/tianran/exp/tfjs/tfjs_example/model/model.json');
  return model
}

async function app() {
  // Create an object from Tensorflow.js data API which could capture image 
  // from the web camera as Tensor.
  const webcam = await tf.data.webcam(webcamElement);
  console.log('Loading mobilenet..');

  // Load the model.
  // net = await mobilenet.load();
  net = await loadModel()
  //warm up 
  // dummy_input = tf.cast(tf.zeros([1, 224, 224, 3]), 'int32');
  dummy_input = tf.zeros([1, 256, 256, 3])
  net.predict(dummy_input)

  // tf.tidy(() => net.predict(webcam.capture()));
  console.log('Successfully loaded model');
  
  
  while (true) {
    const img = await webcam.capture();
    // const result = await net.classify(img);
    // const t4d = tf.cast(tf.tensor4d(Array.from(img.dataSync()),[1,224,224,3]), 'int32');
    // console.time("dataSync")
    // img1 = img.dataSync()
    // console.timeEnd("dataSync")
    // console.time("cpu2tensor")
    // const t4d = tf.tensor4d(Array.from(img1),[1,256,256,3])
    // console.timeEnd("cpu2tensor")
    console.time("normalize")
    t4d_norm = tf.tidy(() => img.expandDims(0).toFloat().div(255))
    // t4d_norm = tf.tidy(() => img.expandDims(0))
    console.timeEnd("normalize")
    var t0 = performance.now();
    console.time("predict");
    const result = await net.predict(t4d_norm);
    console.timeEnd("predict");
    var t1 = performance.now();
    document.getElementById('console').innerText = "inference took " + (t1 - t0) + " milliseconds.";
    // console.log("inference took " + (t1 - t0) + " milliseconds.");
    // document.getElementById('console').innerText = result;
    // document.getElementById('console').innerText = `
    //   prediction: ${result[0].className}\n
    //   probability: ${result[0].probability}
    // `;
    console.time("get result");
    // resultData = await result.data();
    resultData = await result.data();
    console.timeEnd("get result");
    console.time("fp32_uint8");
    bytes = new Uint8Array(resultData.length)
    for (let i = 0; i < bytes.length; ++i) {
      bytes[i] = resultData[i] * 255;
    }
    console.timeEnd("fp32_uint8");
    // console.log(resultData)
    // Dispose the tensor to release the memory.
    img.dispose();

    // Give some breathing room by waiting for the next animation frame to
    // fire.
    await tf.nextFrame();
  }
}

// loadModel();
app();
