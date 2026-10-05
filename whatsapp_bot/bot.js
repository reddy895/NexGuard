/**
 * NexGuard WhatsApp Bot — QR Code Login
 * =======================================
 * Uses system Chrome/Chromium (no Puppeteer download needed).
 * Starts HTTP server on port 3001 for the Python pipeline to call.
 *
 * On first run: prints QR code in terminal — scan with WhatsApp on your phone.
 * Session is saved to .wwebjs_auth/ so you only scan once.
 *
 * Endpoints:
 *   GET  /health  → {"ok": true, "status": "READY"|...}
 *   GET  /status  → {"status": "READY"|"WAITING_QR"|"LOADING"|"DISCONNECTED"}
 *   POST /send    → Body: {"phone": "9591152862", "message": "..."} → {"ok": true}
 */

const { Client, LocalAuth } = require('whatsapp-web.js');
const qrcode = require('qrcode-terminal');
const express = require('express');
const http = require('http');

const PORT = process.env.WHATSAPP_PORT || 3001;

// Detect system Chrome/Chromium
const fs = require('fs');
const CHROME_PATHS = [
    '/usr/bin/google-chrome',
    '/usr/bin/google-chrome-stable',
    '/usr/bin/chromium-browser',
    '/usr/bin/chromium',
    '/usr/local/bin/chromium',
];
let CHROME_EXEC = CHROME_PATHS.find(p => {
    try { return fs.existsSync(p); } catch { return false; }
});

if (!CHROME_EXEC) {
    console.error('[WhatsApp Bot] ERROR: No Chrome/Chromium found. Install with:');
    console.error('  sudo apt-get install -y chromium-browser');
    process.exit(1);
}
console.log(`[WhatsApp Bot] Using browser: ${CHROME_EXEC}`);

// ─── State ────────────────────────────────────────────────────────────────────
let botStatus = 'LOADING';
let client = null;

// ─── Express HTTP Server ──────────────────────────────────────────────────────
const app = express();
app.use(express.json());

app.get('/health', (req, res) => {
    res.json({ ok: true, status: botStatus });
});

app.get('/status', (req, res) => {
    res.json({ status: botStatus });
});

app.post('/send', async (req, res) => {
    const { phone, message } = req.body || {};

    if (!phone || !message) {
        return res.status(400).json({ ok: false, error: 'phone and message are required' });
    }

    if (botStatus !== 'READY') {
        return res.status(503).json({
            ok: false,
            error: `WhatsApp not ready — status: ${botStatus}. Scan the QR code first.`
        });
    }

    try {
        const digits = phone.replace(/[^0-9]/g, '');
        const chatId = digits + '@c.us';
        await client.sendMessage(chatId, message);
        console.log(`[WhatsApp] ✓ Message sent to +${digits}`);
        res.json({ ok: true, phone: digits });
    } catch (err) {
        console.error(`[WhatsApp] Send error: ${err.message}`);
        res.status(500).json({ ok: false, error: err.message });
    }
});

// ─── WhatsApp Client ──────────────────────────────────────────────────────────
function startClient() {
    client = new Client({
        authStrategy: new LocalAuth({ dataPath: '.wwebjs_auth' }),
        puppeteer: {
            executablePath: CHROME_EXEC,
            headless: true,
            args: [
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-accelerated-2d-canvas',
                '--disable-gpu',
                '--no-first-run',
                '--no-zygote',
                '--single-process'
            ]
        }
    });

    client.on('qr', (qr) => {
        botStatus = 'WAITING_QR';
        console.log('\n' + '═'.repeat(62));
        console.log('  NEXGUARD WHATSAPP — SCAN THIS QR CODE WITH YOUR PHONE');
        console.log('  Open WhatsApp → ⋮ → Linked Devices → Link a Device');
        console.log('═'.repeat(62) + '\n');
        qrcode.generate(qr, { small: true });
        console.log('\n  Waiting for QR scan...\n');
    });

    client.on('loading_screen', (percent) => {
        process.stdout.write(`\r[WhatsApp] Loading: ${percent}%   `);
    });

    client.on('authenticated', () => {
        console.log('\n[WhatsApp] Authenticated successfully — session saved.');
        botStatus = 'LOADING';
    });

    client.on('ready', () => {
        botStatus = 'READY';
        const info = client.info || {};
        const name = info.pushname || 'Unknown';
        const number = (info.wid || {}).user || 'Unknown';
        console.log('\n' + '═'.repeat(62));
        console.log('  WHATSAPP BOT READY');
        console.log(`  Logged in as : ${name} (+${number})`);
        console.log(`  HTTP API     : http://127.0.0.1:${PORT}`);
        console.log('═'.repeat(62) + '\n');
    });

    client.on('auth_failure', (msg) => {
        console.error(`[WhatsApp] Auth failure: ${msg}`);
        botStatus = 'DISCONNECTED';
        setTimeout(startClient, 5000);
    });

    client.on('disconnected', (reason) => {
        console.warn(`[WhatsApp] Disconnected: ${reason}`);
        botStatus = 'DISCONNECTED';
        setTimeout(startClient, 5000);
    });

    client.initialize().catch(err => {
        console.error(`[WhatsApp] Init error: ${err.message}`);
        botStatus = 'DISCONNECTED';
    });
}

// ─── Start ────────────────────────────────────────────────────────────────────
const server = http.createServer(app);
server.listen(PORT, '127.0.0.1', () => {
    console.log(`[WhatsApp Bot] HTTP server listening on port ${PORT}`);
    startClient();
});

// Graceful shutdown
async function shutdown() {
    console.log('\n[WhatsApp Bot] Shutting down...');
    if (client) {
        try { await client.destroy(); } catch {}
    }
    server.close();
    process.exit(0);
}

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
