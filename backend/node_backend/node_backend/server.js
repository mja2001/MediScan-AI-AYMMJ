const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const redis = require('redis');

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

const client = redis.createClient();

wss.on('connection', (ws) => {
  console.log('Client connected');
  ws.on('message', (message) => {
    // Broadcast real-time updates
    wss.clients.forEach(client => client.send(message));
  });
});

app.get('/', (req, res) => res.send('Node Backend Running'));

server.listen(3000, () => console.log('Listening on 3000'));
