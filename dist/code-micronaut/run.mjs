#!/usr/bin/env node
import http from 'http';
import crypto from 'crypto';

const PORT = Number(process.env.PORT || 3215);
const state = { startedAt: Date.now(), requests: 0, compileRequests: 0, compileErrors: 0, latencies: [] };

function avg(values) {
  if (values.length === 0) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function parseBody(req) {
  return new Promise((resolve, reject) => {
    let raw = '';
    req.on('data', (chunk) => {
      raw += chunk;
      if (raw.length > 1_000_000) reject(new Error('Body too large'));
    });
    req.on('end', () => {
      if (!raw.trim()) return resolve({});
      try { resolve(JSON.parse(raw)); } catch { reject(new Error('Invalid JSON')); }
    });
    req.on('error', reject);
  });
}

function send(res, code, payload) {
  res.writeHead(code, { 'content-type': 'application/json' });
  res.end(JSON.stringify(payload));
}

const server = http.createServer(async (req, res) => {
  const start = Date.now();
  state.requests += 1;

  try {
    if (req.method === 'GET' && req.url === '/health') {
      return send(res, 200, { service: 'code-micronaut', status: 'ok', port: PORT });
    }

    if (req.method === 'GET' && req.url === '/metrics') {
      const uptimeSec = Math.max(1, (Date.now() - state.startedAt) / 1000);
      return send(res, 200, {
        requests_total: state.requests,
        compile_requests_total: state.compileRequests,
        compile_errors_total: state.compileErrors,
        rps: Number((state.requests / uptimeSec).toFixed(2)),
        avg_latency_ms: Number(avg(state.latencies).toFixed(2))
      });
    }

    if (req.method === 'POST' && req.url === '/compile') {
      const body = await parseBody(req);
      const source = String(body.source || '');
      const language = String(body.language || 'plaintext');

      if (!source.trim()) {
        state.compileErrors += 1;
        return send(res, 400, { error: 'source is required' });
      }

      state.compileRequests += 1;
      const artifactId = crypto.createHash('sha256').update(language + ':' + source).digest('hex').slice(0, 16);
      return send(res, 200, {
        ok: true,
        language,
        artifact_id: artifactId,
        byte_size: Buffer.byteLength(source, 'utf8'),
        warnings: source.length < 20 ? ['source is very short'] : []
      });
    }

    return send(res, 404, { error: 'not_found' });
  } catch (err) {
    state.compileErrors += 1;
    return send(res, 400, { error: err.message });
  } finally {
    const latency = Date.now() - start;
    state.latencies.push(latency);
    if (state.latencies.length > 200) state.latencies.shift();
  }
});

server.listen(PORT, '0.0.0.0', () => {
  console.log(`CODE-1 listening on :${PORT}`);
});
