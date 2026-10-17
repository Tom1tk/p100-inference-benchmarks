'use strict';

const path = require('path');
const express = require('express');

const PORT = 4001;
const HOST = '0.0.0.0';

const app = express();

// Serve the single-page site (index.html) for any non-API route.
app.use(express.static(path.join(__dirname, 'public')));

app.get('/', (_req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Simple health check for monitoring.
app.get('/healthz', (_req, res) => {
  res.json({ status: 'ok', name: 'p5-fork-opencode', uptime: process.uptime() });
});

app.listen(PORT, HOST, () => {
  console.log(`p5-fork-opencode listening on http://localhost:${PORT}`);
});
