import React from 'react'
import * as tf from '@tensorflow/tfjs';

export default function Tf() {
    let trainingData = [];
        const labels = [
            "None",
            "✊ (Rock)",
            "🖐 (Paper)",
            "✌️ (Scissors)",
            "👌 (Letter D)",
            "👍 (Thumb Up)",
            "🖖 (Vulcan)",
            "🤟 (ILY - I Love You)"
        ];

        function setText( text ) {
            document.getElementById( "status" ).innerText = text;
        }

        async function predictImage() {
            if( !hasTrained ) { return; } // Skip prediction until trained
            const img = await getWebcamImage();
            
            let result = tf.tidy( () => {
                const input = img.reshape( [ 1, 224, 224, 3 ] );
                return model.predict( input );
            });
            img.dispose();
            let prediction = await result.data();
            console.log(prediction)
            result.dispose();
            // Get the index of the highest value in the prediction
            let id = prediction.indexOf( Math.max( ...prediction ) );
            console.log(id)
            setText( labels[id] );
        }

        function createTransferModel( model ) {
            // Create the truncated base model (remove the "top" layers, classification + bottleneck layers)
            const bottleneck = model.getLayer( "dropout" ); // This is the final layer before the conv_pred pre-trained classification layer
            const baseModel = tf.model({
                inputs: model.inputs,
                outputs: bottleneck.output
            });
            // Freeze the convolutional base
            for( const layer of baseModel.layers ) {
                layer.trainable = false;
            }
            // Add a classification head
            const newHead = tf.sequential();
            newHead.add( tf.layers.flatten( {
                inputShape: baseModel.outputs[ 0 ].shape.slice( 1 )
            } ) );
            newHead.add( tf.layers.dense( { units: 100, activation: 'relu' } ) );
            newHead.add( tf.layers.dense( { units: 100, activation: 'relu' } ) );
            newHead.add( tf.layers.dense( { units: 10, activation: 'relu' } ) );
            newHead.add( tf.layers.dense( {
                units: labels.length,
                kernelInitializer: 'varianceScaling',
                useBias: false,
                activation: 'softmax'
            } ) );
            // Build the new model
            const newOutput = newHead.apply( baseModel.outputs[ 0 ] );
            const newModel = tf.model( { inputs: baseModel.inputs, outputs: newOutput } );
            return newModel;
        }

        async function trainModel() {
            hasTrained = false;
            setText( "Training..." ); 
            // Setup training data
            const imageSamples = [];
            const targetSamples = [];
            for(let i=0;i<trainingData.length;i++){
                let analyzeImage = await trainingData[i].image
                console.log(analyzeImage)
                imageSamples.push( analyzeImage );
                let cat = [];
                for( let c = 0; c < labels.length; c++ ) {
                    cat.push( c === trainingData[i].category ? 1 : 0 );
                }
                targetSamples.push( tf.tensor1d( cat ) );
            }
            console.log(trainingData)
            // trainingData.forEach( async (sample) => {
            //         let analyzeImage = await sample.image
            //         console.log(analyzeImage)
            //         imageSamples.push( analyzeImage );
            //         let cat = [];
            //         for( let c = 0; c < labels.length; c++ ) {
            //             cat.push( c === sample.category ? 1 : 0 );
            //         }
            //         targetSamples.push( tf.tensor1d( cat ) );
            //         console.log(tf.stack( imageSamples ))
            // });
            console.log(imageSamples)
            const xs = tf.stack( imageSamples );
            const ys = tf.stack( targetSamples );
            // Train the model on new image samples
            model.compile( { loss: "meanSquaredError", optimizer: "adam", metrics: [ "acc" ] } );

            await model.fit( xs, ys, {
                epochs: 30,
                shuffle: true,
                callbacks: {
                    onEpochEnd: ( epoch, logs ) => {
                        console.log( "Epoch #", epoch, logs );
                    }
                }
            });
            console.log('hasTrained')
            hasTrained = true;

        }

        // Mobilenet v1 0.25 224x224 model
        const mobilenet = "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json";

        let model = null;
        let hasTrained = false;

        async function setupWebcam() {
            return new Promise( ( resolve, reject ) => {
                const webcamElement = document.getElementById( "webcam" );
                const navigatorAny = navigator;
                navigator.getUserMedia = navigator.getUserMedia ||
                navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia ||
                navigatorAny.msGetUserMedia;
                if( navigator.getUserMedia ) {
                    navigator.getUserMedia( { video: true },
                        stream => {
                            webcamElement.srcObject = stream;
                            webcamElement.addEventListener( "loadeddata", resolve, false );
                        },
                    error => reject());
                }
                else {
                    reject();
                }
            });
        }

        async function getWebcamImage() {
            const img = ( await webcam.capture() ).toFloat();
            const normalized = img.div( 127 ).sub( 1 );
            return normalized;
        }

        function captureSample( category ) {
            return (()=>{
                trainingData.push( {
                    image:  getWebcamImage(),
                    category: category
                });
                setText( "Captured: " + labels[ category ] );
            })  
        }
        let webcam = null;
        (async () => {
            // Load the model
            model = await tf.loadLayersModel( mobilenet );
            model = createTransferModel( model );
            await setupWebcam();
            webcam = await tf.data.webcam( document.getElementById( "webcam" ) );
            // Setup prediction every 200 ms
            setInterval( predictImage, 200 );
        })();

    return (
        <div>
            <video autoPlay playsInline muted id="webcam" width="224" height="224"></video>
            <div id="buttons">
                <button onClick={captureSample(0)}>None</button>
                <button onClick={captureSample(1)}>✊ (Rock)</button>
                <button onClick={captureSample(2)}>🖐 (Paper)</button>
                <button onClick={captureSample(3)}>✌️ (Scissors)</button>
                <button onClick={captureSample(4)}>👌 (Letter D)</button>
                <button onClick={captureSample(5)}>👍 (Thumb Up)</button>
                <button onClick={captureSample(6)}>🖖 (Vulcan)</button>
                <button onClick={captureSample(7)}>🤟 (ILY - I Love You)</button>
                <button onClick={trainModel}>Train</button>
            </div>
            <h1 id="status">Loading...</h1>
        </div>
    );
}
