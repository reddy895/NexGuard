const { Client, LocalAuth } = require('whatsapp-web.js');
const qrcode = require('qrcode-terminal');
const path = require('path');
const fs = require('fs');
const http = require('http');

const authPath = path.join(__dirname, '.wwebjs_auth');
const PORT = process.env.WHATSAPP_PORT || 3001;

// Puppeteer launch options
const puppeteerOptions = {
  headless: true,
  executablePath: process.env.CHROME_PATH || '/usr/bin/google-chrome',
  args: [
    '--no-sandbox',
    '--disable-setuid-sandbox',
    '--disable-dev-shm-usage',
    '--disable-accelerated-2d-canvas',
    '--no-first-run',
    '--no-zygote',
    '--single-process',
    '--disable-gpu'
  ]
};

if (!fs.existsSync(puppeteerOptions.executablePath)) {
  delete puppeteerOptions.executablePath;
}

const command = process.argv[2] || 'listen';
const targetNumber = process.argv[3];
const messageBody = process.argv[4];

const client = new Client({
  authStrategy: new LocalAuth({ dataPath: authPath }),
  puppeteer: puppeteerOptions
});

let isReady = false;

// Handle QR code generation
client.on('qr', (qr) => {
  if (command === 'qr' || command === 'setup') {
    console.log('\n==================================================');
    console.log('            NEXGUARD WHATSAPP SETUP');
    console.log('==================================================');
    console.log('Scan the QR code below using WhatsApp:');
    console.log('WhatsApp -> Linked Devices -> Link a Device\n');
    qrcode.generate(qr, { small: true });
    console.log('==================================================\n');
  } else {
    console.log('STATUS: AUTHENTICATING');
    if (command === 'status') {
      process.exit(0);
    }
  }
});

// Incoming message listener
client.on('message', async (msg) => {
  const text = msg.body.trim();
  console.log(`[WHATSAPP LISTEN] Received message from ${msg.from}: ${text}`);

  if (text.toLowerCase() === '!ping') {
    await msg.reply('🏓 *NexGuard Bot:* Pong! System is online and monitoring CCTV stream (Max 15 FPS).');
  } else if (text.toLowerCase() === '!status') {
    await msg.reply('🟢 *NexGuard Status:* Active & Monitoring.\n- Max FPS: 15.0\n- YOLO Model: Ready\n- WhatsApp Integration: Active');
  } else if (text.toLowerCase() === '!help') {
    await msg.reply('ℹ️ *NexGuard Bot Commands:*\n- `!status` - Check AI CCTV system status\n- `!ping` - Test bot response\n- `!alert` - Request latest incident report');
  }
});

client.on('ready', async () => {
  isReady = true;
  console.log('STATUS: CONNECTED');

  if (command === 'status') {
    await client.destroy();
    process.exit(0);
  } else if (command === 'send') {
    if (!targetNumber || !messageBody) {
      console.log('ERROR: Missing recipient number or message string');
      await client.destroy();
      process.exit(1);
    }

    try {
      const sanitizedNumber = targetNumber.replace(/[^0-9]/g, '');
      const chatId = `${sanitizedNumber}@c.us`;
      await client.sendMessage(chatId, messageBody);
      console.log('RESULT: SUCCESS');
    } catch (err) {
      console.log(`RESULT: ERROR - ${err.message}`);
    }
    await client.destroy();
    process.exit(0);
  } else if (command === 'listen') {
    console.log(`[WHATSAPP BOT] Listening for incoming messages & starting HTTP IPC server on port ${PORT}...`);
    
    // Start local HTTP server to receive instant dispatch requests from Python
    const server = http.createServer((req, res) => {
      if (req.method === 'POST' && req.url === '/send') {
        let body = '';
        req.on('data', chunk => { body += chunk.toString(); });
        req.on('end', async () => {
          try {
            const data = JSON.parse(body);
            const sanitized = (data.number || '').replace(/[^0-9]/g, '');
            if (!sanitized || !data.message) {
              res.writeHead(400, { 'Content-Type': 'application/json' });
              return res.end(JSON.stringify({ success: false, error: 'Invalid parameters' }));
            }
            const chatId = `${sanitized}@c.us`;
            await client.sendMessage(chatId, data.message);
            console.log(`[WHATSAPP DISPATCH] Successfully sent message to ${sanitized}`);
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ success: true }));
          } catch (err) {
            console.error(`[WHATSAPP ERROR] ${err.message}`);
            res.writeHead(500, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ success: false, error: err.message }));
          }
        });
      } else {
        res.writeHead(404, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Not found' }));
      }
    });

    server.listen(PORT, () => {
      console.log(`[WHATSAPP HTTP SERVER] Listening on http://localhost:${PORT}`);
    });
  }
});

client.on('auth_failure', async (msg) => {
  console.log(`STATUS: AUTH_FAILURE - ${msg}`);
  process.exit(1);
});

client.on('disconnected', (reason) => {
  console.log(`STATUS: DISCONNECTED - ${reason}`);
  process.exit(0);
});

// Timeout safeguard for non-listening commands
setTimeout(() => {
  if (!isReady && command === 'status') {
    console.log('STATUS: NOT CONNECTED');
    process.exit(0);
  }
}, 8000);

client.initialize().catch((err) => {
  console.log(`STATUS: ERROR - ${err.message}`);
  process.exit(1);
});
