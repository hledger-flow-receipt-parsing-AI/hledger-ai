// Sidecar of hledger-ai for the qvac SDK (https://qvac.tether.io).
//
// Reads one JSON request per line on stdin and answers with one line on
// stdout that starts with MARK (the SDK writes its own log lines to stdout
// too). Only local models are used; hledger-ai starts this process without
// network access once the models are downloaded.
import { close, loadModel, ocr, OCR_LATIN, unloadModel } from '@qvac/sdk';
import readline from 'node:readline';

const MARK = '@@hledger-ai@@ ';
let ocrModel = null;

function reply(message) {
  process.stdout.write(MARK + JSON.stringify(message) + '\n');
}

async function ocrModelId() {
  if (ocrModel === null) {
    ocrModel = await loadModel({
      modelSrc: OCR_LATIN,
      modelConfig: {
        langList: ['nl', 'en', 'de', 'fr', 'es'],
        magRatio: 1.5,
        contrastRetry: false,
        lowConfidenceThreshold: 0.5,
        recognizerBatchSize: 1,
      },
    });
  }
  return ocrModel;
}

const lines = readline.createInterface({ input: process.stdin });
for await (const line of lines) {
  let request;
  try {
    request = JSON.parse(line);
  } catch {
    continue;
  }
  try {
    if (request.op === 'close') break;
    if (request.op === 'ocr') {
      const { blocks } = ocr({
        modelId: await ocrModelId(),
        image: request.image,
        options: { paragraph: false },
      });
      reply({ id: request.id, blocks: await blocks });
    } else if (request.op === 'warmup') {
      await ocrModelId();
      reply({ id: request.id, ok: true });
    } else {
      reply({ id: request.id, error: `unknown op ${request.op}` });
    }
  } catch (error) {
    reply({ id: request.id, error: String((error && error.message) || error) });
  }
}
try {
  if (ocrModel !== null) await unloadModel({ modelId: ocrModel, clearStorage: false });
  await close();
} finally {
  process.exit(0);
}
