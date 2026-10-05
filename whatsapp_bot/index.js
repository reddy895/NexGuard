const { Client, LocalAuth } = require('whatsapp-web.js');
const qrcode = require('qrcode-terminal');
const path = require('path');
const fs = require('fs');

const authPath = path.join(__dirname, '.wwebjs_auth');

// Options for puppeteer
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

// Check system chrome binary fallback
if (!fs.existsSync(puppeteerOptions.executablePath)) {
  delete puppeteerOptions.executablePath;
}

const command = process.argv[2] || 'status';
const targetNumber = process.argv[3];
const messageBody = process.argv[4];

const client = new Client({
  authStrategy: new LocalAuth({ dataPath: authPath }),
  puppeteer: puppeteerOptions
});

let isReady = false;

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
    process.exit(0);
  }
});

client.on('ready', async () => {
  isReady = true;

  if (command === 'status' || command === 'qr' || command === 'setup') {
    console.log('STATUS: CONNECTED');
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
  }
});

client.on('auth_failure', async (msg) => {
  console.log(`STATUS: AUTH_FAILURE - ${msg}`);
  await client.destroy();
  process.exit(1);
});

client.on('disconnected', (reason) => {
  console.log(`STATUS: DISCONNECTED - ${reason}`);
  process.exit(0);
});

// Timeout safeguard
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
