'use strict';

const express = require('express');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 4000;
const HOST = process.env.HOST || '127.0.0.1';

app.use(express.static(path.join(__dirname, 'public')));

app.get('/', (_req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.get('/healthz', (_req, res) => {
  res.json({ status: 'ok', service: 'p5-fork-serve', uptime: process.uptime() });
});

const server = app.listen(PORT, HOST, () => {
  console.log(`p5-fork-serve listening on http://${HOST}:${PORT}`);
});

process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down');
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(0), 3000).unref();
});
