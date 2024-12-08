const predictClassification = require('../services/inferenceService');
const crypto = require('crypto');
const storeData = require('../services/storeData');


async function postPredictHandler(request, h) {
  const { image } = request.payload;
  const { model } = request.server.app;

  try {
    const { confidenceScore, label, suggestion } = await predictClassification(model, image);
    const id = crypto.randomUUID();
    const createdAt = new Date().toISOString();

    const data = {
      "id": id,
      "result": label,
      "suggestion": suggestion,
      "createdAt": createdAt
    };

    await storeData(id, data);


    // Response untuk prediksi
    
    const response = h.response({
      status: 'success',
      message: confidenceScore > 99 ? 'Model is predicted successfully' : 'Model is predicted successfully but under threshold.',
      data
    });
    response.code(201);
    return response;
  } catch (error) {
    return h.response({
      status: 'fail',
      message: 'Terjadi kesalahan dalam melakukan prediksi'
    }).code(500);
  }
}

module.exports = postPredictHandler;
