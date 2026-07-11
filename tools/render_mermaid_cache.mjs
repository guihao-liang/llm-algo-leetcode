#!/usr/bin/env node
import crypto from 'node:crypto'
import http from 'node:http'
import { existsSync, mkdirSync } from 'node:fs'
import { readFile, readdir, writeFile, mkdtemp } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const ROOT = path.resolve(fileURLToPath(new URL('..', import.meta.url)))
const DOCS = path.join(ROOT, 'docs')
const OUTPUT = path.join(DOCS, '.vitepress', 'cache', 'mermaid-cache.json')
const MERMAID_DIST = path.join(DOCS, 'node_modules', 'mermaid', 'dist')
const CHROME = process.env.CHROME_BIN || 'google-chrome'

const MERMAID_FENCE = /^```mermaid\s*$/
const FENCE_END = /^```\s*$/

const walkMarkdownFiles = async () => {
  const files = [path.join(ROOT, 'README.md'), path.join(DOCS, 'index.md')]
  const stack = [DOCS]

  while (stack.length > 0) {
    const current = stack.pop()
    for (const entry of await readdir(current, { withFileTypes: true })) {
      if (entry.isDirectory()) {
        if (entry.name === 'node_modules') continue
        stack.push(path.join(current, entry.name))
      } else if (entry.isFile() && entry.name.endsWith('.md')) {
        files.push(path.join(current, entry.name))
      }
    }
  }

  return [...new Set(files)].sort()
}

const extractBlocks = async (filePath) => {
  const lines = (await readFile(filePath, 'utf8')).split(/\r?\n/)
  const blocks = []
  let inBlock = false
  let startLine = 0
  let buffer = []

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i]
    if (!inBlock && MERMAID_FENCE.test(line)) {
      inBlock = true
      startLine = i + 2
      buffer = []
      continue
    }
    if (inBlock && FENCE_END.test(line)) {
      blocks.push({ startLine, content: buffer.join('\n') })
      inBlock = false
      buffer = []
      continue
    }
    if (inBlock) buffer.push(line)
  }

  return blocks
}

const sha256 = (text) => crypto.createHash('sha256').update(text, 'utf8').digest('hex')

const buildChromeHtml = (payload, mermaidBaseUrl) => {
  const encoded = encodeURIComponent(JSON.stringify(payload))
  return `<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Mermaid cache renderer</title>
  </head>
  <body>
    <textarea id="input" hidden>${encoded}</textarea>
    <script type="module">
      document.body.setAttribute('data-phase', 'boot')
      const input = JSON.parse(decodeURIComponent(document.getElementById('input').textContent || '{}'))
      document.body.setAttribute('data-phase', 'before-import')
      const mermaid = (await import(${JSON.stringify(`${mermaidBaseUrl}/mermaid.core.mjs`)})).default
      document.body.setAttribute('data-phase', 'after-import')
      mermaid.initialize({
        startOnLoad: false,
        securityLevel: 'strict',
        theme: 'default',
        flowchart: { htmlLabels: false }
      })

      try {
        const results = {}
        const entries = Object.entries(input)
        for (const [hash, text] of entries) {
          document.body.setAttribute('data-current', hash)
          const { svg } = await mermaid.render('mermaid_' + hash, text)
          results[hash] = svg
        }
        document.body.setAttribute('data-phase', 'rendered')
        const output = document.createElement('textarea')
        output.id = 'results'
        output.textContent = encodeURIComponent(JSON.stringify(results))
        document.body.replaceChildren(output)
      } catch (error) {
        document.body.setAttribute('data-phase', 'error')
        const output = document.createElement('textarea')
        output.id = 'error'
        output.textContent = encodeURIComponent(String(error && error.stack ? error.stack : error))
        document.body.replaceChildren(output)
      }
    </script>
  </body>
  </html>`
}

const contentTypeFor = (filePath) => {
  if (filePath.endsWith('.mjs') || filePath.endsWith('.js')) return 'application/javascript; charset=utf-8'
  if (filePath.endsWith('.css')) return 'text/css; charset=utf-8'
  if (filePath.endsWith('.svg')) return 'image/svg+xml; charset=utf-8'
  if (filePath.endsWith('.json')) return 'application/json; charset=utf-8'
  if (filePath.endsWith('.html')) return 'text/html; charset=utf-8'
  return 'application/octet-stream'
}

const startServer = async (htmlPath) => {
  let server
  const port = await new Promise((resolve, reject) => {
    server = http.createServer(async (req, res) => {
      try {
        const url = new URL(req.url || '/', 'http://127.0.0.1')
        if (url.pathname === '/' || url.pathname === '/index.html') {
          res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' })
          res.end(await readFile(htmlPath, 'utf8'))
          return
        }
        if (url.pathname.startsWith('/mermaid-dist/')) {
          const filePath = path.join(MERMAID_DIST, url.pathname.slice('/mermaid-dist/'.length))
          if (!filePath.startsWith(MERMAID_DIST)) {
            res.writeHead(403)
            res.end('Forbidden')
            return
          }
          const data = await readFile(filePath)
          res.writeHead(200, { 'Content-Type': contentTypeFor(filePath) })
          res.end(data)
          return
        }
        res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' })
        res.end('Not found')
      } catch (error) {
        res.writeHead(500, { 'Content-Type': 'text/plain; charset=utf-8' })
        res.end(String(error))
      }
    })
    server.once('error', reject)
    server.listen(0, '127.0.0.1', () => {
      const address = server.address()
      if (!address || typeof address === 'string') {
        reject(new Error('Failed to start local Mermaid renderer server'))
        return
      }
      resolve(address.port)
    })
  })

  return {
    port,
    close: async () =>
      await new Promise((resolve) => {
        server.close(() => resolve(undefined))
      }),
  }
}

const renderInChrome = async (payload) => {
  const tempDir = await mkdtemp(path.join(os.tmpdir(), 'mermaid-cache-'))
  const htmlPath = path.join(tempDir, 'index.html')
  const { port, close } = await startServer(htmlPath)
  const baseUrl = `http://127.0.0.1:${port}`
  await writeFile(htmlPath, buildChromeHtml(payload, `${baseUrl}/mermaid-dist`), 'utf8')

  try {
    const result = spawnSync(
      CHROME,
      [
        '--headless=new',
        '--no-sandbox',
        '--disable-gpu',
        '--disable-dev-shm-usage',
        '--virtual-time-budget=120000',
        '--dump-dom',
        `${baseUrl}/index.html`,
      ],
      { encoding: 'utf8' }
    )

    if (result.error) {
      throw result.error
    }
    if (result.status !== 0) {
      throw new Error((result.stderr || result.stdout || 'Chrome failed').trim())
    }

    const dom = result.stdout
    const errorMatch = dom.match(/<textarea id="error">([\s\S]*?)<\/textarea>/)
    if (errorMatch) {
      throw new Error(decodeURIComponent(errorMatch[1]))
    }
    const resultsMatch = dom.match(/<textarea id="results">([\s\S]*?)<\/textarea>/)
    if (!resultsMatch) {
      throw new Error(
        `Mermaid renderer did not produce results.\nstdout:\n${dom.slice(0, 2000)}\n\nstderr:\n${result.stderr.slice(0, 2000)}`
      )
    }

    return JSON.parse(decodeURIComponent(resultsMatch[1]))
  } finally {
    await close()
  }
}

const main = async () => {
  const files = await walkMarkdownFiles()
  const blocks = []
  for (const filePath of files) {
    const rel = path.relative(ROOT, filePath)
    const fileBlocks = await extractBlocks(filePath)
    for (const block of fileBlocks) {
      blocks.push({ file: rel, ...block })
    }
  }

  const unique = new Map()
  for (const block of blocks) {
    const hash = sha256(block.content)
    if (!unique.has(hash)) {
      unique.set(hash, block.content)
    }
  }

  const rendered = await renderInChrome(Object.fromEntries(unique.entries()))
  const cacheDir = path.dirname(OUTPUT)
  if (!existsSync(cacheDir)) mkdirSync(cacheDir, { recursive: true })
  await writeFile(OUTPUT, JSON.stringify(rendered, null, 2) + '\n', 'utf8')
  process.stdout.write(`Rendered ${Object.keys(rendered).length} Mermaid block(s) into ${path.relative(ROOT, OUTPUT)}\n`)
}

main().catch((error) => {
  process.stderr.write(`${error instanceof Error ? error.stack || error.message : String(error)}\n`)
  process.exit(1)
})
