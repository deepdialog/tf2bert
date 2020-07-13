const tf = require('@tensorflow/tfjs-node');

(async () => {
    const test = tf.tensor([['你', '好']])
    const model = await tf.node.loadSavedModel('../../../bert/albert_base/')
    const p = model.predict(test)
    console.log(p)
})()
