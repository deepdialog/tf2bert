const tf = require('@tensorflow/tfjs-node');

(async () => {
    const x = tf.input({shape: [null, null], dtype: 'string'})
    const bert = await tf.node.loadSavedModel('../../bert')
    const y = bert.predict(x)
    const model = tf.model({inputs: x, outputs, y})
    const test = tf.tensor([['你', '好']])
    const p = model.predict(test)
    console.log(p)
})()
