/**
 * The second text reader: PP-OCRv4 (detection + recognition, ONNX, run by onnxruntime). It reads stylized and thick fonts, and the
 * Chinese / Japanese text, that Tesseract does not. The runtime and the models are given by the caller (onnxruntime-node in the tests,
 * onnxruntime-web in the Decky plugin), so the same code runs in both. The result has the shape of the one of Tesseract (OcrResult).
 */
import type { OcrLine, OcrResult } from './ocr'

interface Ort { Tensor: any; InferenceSession: any }

let ort: Ort | null = null
let det: any = null
let rec: any = null
let keys: string[] = []

export const ppocrStats = { reads: 0, ms: 0, detects: 0, detectMs: 0 }

// How the maps of the detector are turned into boxes
const DET_THRESHOLD = 0.3
const DET_BOX_SCORE = 0.55
const DET_MAX_SIDE = 960
const DET_UNCLIP = 1.6
const REC_HEIGHT = 48
const REC_MAX_WIDTH = 1280

// ---- Words that the recognizer joined ("TAKEIT" for "TAKE IT"): an English word list splits them (the method of the wordninja package, MIT) ----
let wordCost: Map<string, number> | null = null
let longestWord = 0
const SHORT_WORDS = new Set(['a', 'i', 'am', 'an', 'as', 'at', 'be', 'by', 'do', 'go', 'he', 'if', 'in', 'is', 'it', 'me', 'my', 'no', 'of', 'oh', 'ok', 'on', 'or', 'so', 'to', 'up', 'us', 'we'])
// Words further down the list than this are rare: a split into them is not believed
const RARE_RANK = 20000

/** Gives the list of English words (the most common first, one per line); until then nothing is split. */
export function setWordList(text: string | null): void {
    if(text === null){
        wordCost = null
        return
    }
    const words = text.split(/\r?\n/).filter(word => word !== '')
    const scale = Math.log(words.length)
    wordCost = new Map()
    longestWord = 0
    words.forEach((word, rank) => {
        wordCost!.set(word, Math.log((rank + 1) * scale))
        longestWord = Math.max(longestWord, word.length)
    })
}

/** Where a string of letters falls apart into known words; null when it does not (or it is a word as it is). */
function splitWords(letters: string): number[] | null {
    if(wordCost === null || letters.length < 6 || letters.length > 40){
        return null
    }
    const lower = letters.toLowerCase()
    if(wordCost.has(lower)){
        return null
    }
    const n = lower.length
    const best: number[] = new Array(n + 1).fill(Infinity)
    const from: number[] = new Array(n + 1).fill(0)
    best[0] = 0
    for(let i = 1; i <= n; i++){
        for(let k = 1; k <= Math.min(longestWord, i); k++){
            const piece = lower.slice(i - k, i)
            const cost = wordCost.get(piece)
            // one-letter words are only "a" and "i"
            if(cost === undefined || (k === 1 && piece !== 'a' && piece !== 'i')){
                continue
            }
            if(best[i - k] + cost < best[i]){
                best[i] = best[i - k] + cost
                from[i] = i - k
            }
        }
    }
    if(!isFinite(best[n])){
        return null
    }
    const cuts: number[] = []
    for(let i = n; i > 0; i = from[i]){
        if(from[i] > 0){
            cuts.unshift(from[i])
        }
    }
    if(cuts.length === 0){
        return null
    }
    // A name that is not in the list falls apart into rare little words ("Ozymandias" -> "O zy mandi as"): only common words are believed
    const bounds = [0, ...cuts, n]
    for(let i = 0; i < bounds.length - 1; i++){
        const piece = lower.slice(bounds[i], bounds[i + 1])
        if((piece.length <= 2 && !SHORT_WORDS.has(piece)) || (wordCost.get(piece) ?? Infinity) > Math.log(RARE_RANK * Math.log(wordCost.size))){
            return null
        }
    }
    return cuts
}

/** How much of a reading is made of real words (0..1): a reader that mistakes letters of a pixel font reads "Obtoined o Life Copsule". Without the word list: 1. */
function plausibility(text: string): number {
    if(wordCost === null){
        return 1
    }
    let words = 0
    let known = 0
    for(const token of text.split(/[^A-Za-z']+/)){
        if(token.length < 2){
            continue
        }
        words++
        // names and rare words are not in the list: a word counts when it, or its form without a plural/'s, is known
        const lower = token.toLowerCase().replace(/'s$/, '')
        if(wordCost.has(lower) || wordCost.has(lower.replace(/s$/, '')) || wordCost.has(lower.replace(/(ed|ing)$/, ''))){
            known++
        }
    }
    return words === 0 ? 1 : known / words
}

export // Letters that the reader mixes up in small, pixel-like fonts ("Dycle" for "Cycle", "Seroll" for "Scroll", "Iist" for "list")
const CONFUSED: string[] = ['ceoa', 'od0q', 'li1tj', 'uvnrw', 'hbk', 'mn', 'sz5', 'gq', 'ft', 'yv', 'dc']
const CONFUSED_WITH = new Map<string, string>()
for(const group of CONFUSED){
    for(const letter of group){
        CONFUSED_WITH.set(letter, (CONFUSED_WITH.get(letter) ?? '') + group.replace(letter, ''))
    }
}
// Only words this common are believed as a correction
const COMMON_RANK = 30000

/** A word that is not a word but is one confused letter away from a common word is that word; null when it is a word, a name, or nothing is near. */
function nearWord(token: string): string | null {
    if(wordCost === null || token.length < 4 || token.length > 14){
        return null
    }
    const lower = token.toLowerCase()
    const limit = Math.log(COMMON_RANK * Math.log(wordCost.size))
    if(wordCost.has(lower) || wordCost.has(lower.replace(/s$/, '')) || wordCost.has(lower.replace(/(ed|ing|ly)$/, ''))){
        return null
    }
    let best: string | null = null
    let bestCost = limit
    for(let i = 0; i < lower.length; i++){
        for(const letter of CONFUSED_WITH.get(lower[i]) ?? ''){
            const candidate = lower.slice(0, i) + letter + lower.slice(i + 1)
            const cost = wordCost.get(candidate)
            if(cost !== undefined && cost < bestCost){
                best = candidate
                bestCost = cost
            }
        }
    }
    if(best === null){
        return null
    }
    // the same way of writing as the word that was read
    if(token === token.toUpperCase()){
        return best.toUpperCase()
    }
    // a capital I that is really a small l ("Iist") is not a capital
    return token[0] === token[0].toUpperCase() && !(token[0] === 'I' && best[0] !== 'i') ? best[0].toUpperCase() + best.slice(1) : best
}

/** Every word of a line that the reader got wrong by a confused letter is put right (a word that is a word, or a name nothing is near to, is left as it is). */
export function fixNearWords(text: string): string {
    if(wordCost === null){
        return text
    }
    return text.replace(/[A-Za-z]{4,14}/g, (word, offset: number) => (offset > 0 && text[offset - 1] === "'") ? word : (nearWord(word) ?? word))
}

export function splitJoinedWords(text: string): string {
    if(wordCost === null){
        return text
    }
    // A full stop or a quote that the reader did not follow with a space ("skill.Iron", "'Iron'skill"), then every long run of letters that is not a word
    const spaced = text.replace(/([a-z])([.!?;:,])([A-Z])/g, '$1$2 $3').replace(/([a-z])'([A-Z][A-Za-z]+)'/g, "$1 '$2'").replace(/([A-Za-z]),([a-z]{2,})/g, '$1, $2').replace(/([a-z]s)'([a-z]{3,})/g, "$1' $2").replace(/'([A-Za-z]+)'(?=[A-Za-z])/g, "'$1' ")
    return spaced.replace(/[A-Za-z]{6,40}/g, (run) => {
        const cuts = splitWords(run)
        if(cuts === null){
            return run
        }
        let out = ''
        let last = 0
        for(const cut of cuts){
            out += run.slice(last, cut) + ' '
            last = cut
        }
        return out + run.slice(last)
    })
}

export function ppocrReady(): boolean {
    return (det !== null && rec !== null) || remoteReady
}

// ---- The reader in a worker of its own ----
// The models run in WebAssembly, in the thread that calls them: a reading of a paragraph keeps that thread busy for a second or two, and a worker that reads cannot look at the
// picture meanwhile (a plate that is stale stays). So the models can live in a second worker: `serveReader` runs there, `useRemoteReader` connects the engine to it by a channel.
let remote: MessagePort | null = null
let remoteReady = false
let remoteId = 0
let remoteFailures = 0
let remoteOnFailed: (() => void) | null = null
const remoteWaiting = new Map<number, (answer: any) => void>()
// The longest an answer may take (a reading is 1-3 s on the Deck, the models are loaded once)
let REMOTE_TIMEOUT_MS = 20000
const REMOTE_MAX_FAILURES = 3
export const remoteStats = { calls: 0, failures: 0 }
/** For tests: how long an answer may take. */
export function setRemoteTimeout(ms: number): void {
    REMOTE_TIMEOUT_MS = ms
}

/** The engine side: the reader is in the worker at the other end of the port. `onReady` is called when it has its models, `onFailed` when it does not answer (the caller reads itself then). */
export function useRemoteReader(port: MessagePort, onReady: () => void, onFailed: () => void): void {
    remote = port
    remoteReady = false
    remoteFailures = 0
    remoteOnFailed = onFailed
    port.onmessage = (event: MessageEvent) => {
        const answer = event.data
        if(answer.type === 'ready'){
            remoteReady = true
            onReady()
        } else if(answer.type === 'failed'){
            giveUpRemote()
        } else if(answer.id !== undefined){
            const done = remoteWaiting.get(answer.id)
            if(done !== undefined){
                remoteWaiting.delete(answer.id)
                done(answer)
            }
        }
    }
}

/** The reader of the other worker is given up (it did not come up in time): `onFailed` is called, and the caller reads itself. */
export function abandonRemoteReader(): void {
    giveUpRemote()
}

function giveUpRemote(): void {
    if(remote === null){
        return
    }
    remoteReady = false
    remote = null
    for(const done of Array.from(remoteWaiting.values())){
        done(null)
    }
    remoteWaiting.clear()
    const callback = remoteOnFailed
    remoteOnFailed = null
    callback?.()
}

function askRemote(message: any, transfer: Transferable[]): Promise<any | null> {
    const port = remote
    if(port === null || !remoteReady){
        return Promise.resolve(null)
    }
    remoteStats.calls++
    return new Promise((resolve) => {
        const id = ++remoteId
        const timer = setTimeout(() => {
            remoteWaiting.delete(id)
            remoteStats.failures++
            remoteFailures++
            if(remoteFailures >= REMOTE_MAX_FAILURES){
                giveUpRemote()
            }
            resolve(null)
        }, REMOTE_TIMEOUT_MS)
        remoteWaiting.set(id, (answer: any) => {
            clearTimeout(timer)
            if(answer !== null && answer.error === undefined){
                remoteFailures = 0
            }
            resolve(answer !== null && answer.error === undefined ? answer : null)
        })
        try {
            port.postMessage({ ...message, id }, transfer)
        } catch(error) {
            clearTimeout(timer)
            remoteWaiting.delete(id)
            resolve(null)
        }
    })
}

/** The reader side: answers the requests that come through the port, one at a time (the models are used by one call at a time). */
export function serveReader(port: MessagePort): void {
    let queue: Promise<void> = Promise.resolve()
    port.onmessage = (event: MessageEvent) => {
        const m = event.data
        queue = queue.then(async () => {
            try {
                const rgba = new Uint8ClampedArray(m.buf)
                const started = Date.now()
                if(m.op === 'detect'){
                    const boxes = await detect(rgba, m.w, m.h)
                    port.postMessage({ id: m.id, boxes, ms: Date.now() - started })
                } else if(m.op === 'read'){
                    const result = await recognizeLocal(rgba, m.w, m.h)
                    port.postMessage({ id: m.id, result, ms: Date.now() - started })
                }
            } catch(error) {
                port.postMessage({ id: m.id, error: String(error) })
            }
        })
    }
}

/** Starts the reader. `detModel` / `recModel` are the ONNX files (bytes or a path the runtime can open), `keysText` the dictionary of the recognizer. */
export async function initPpocr(runtime: Ort, detModel: any, recModel: any, keysText: string, options: any = {}): Promise<void> {
    ort = runtime
    det = await runtime.InferenceSession.create(detModel, options)
    rec = await runtime.InferenceSession.create(recModel, options)
    const lines = keysText.split(/\r?\n/)
    if(lines.length > 0 && lines[lines.length - 1] === ''){
        lines.pop()
    }
    keys = ['', ...lines, ' ']
}

export function disposePpocr(): void {
    det = null
    rec = null
    ort = null
}

/** Bilinear resize of a region of an RGBA picture into a float tensor (BGR planes, (x/255 - mean) / std): the way RapidOCR feeds the models. */
function toPlanes(rgba: Uint8ClampedArray, sw: number, sh: number, rx: number, ry: number, rw: number, rh: number, dw: number, dh: number, mean: number[], std: number[]): Float32Array {
    const out = new Float32Array(3 * dw * dh)
    const plane = dw * dh
    for(let y = 0; y < dh; y++){
        const fy = Math.min(rh - 1, Math.max(0, (y + 0.5) * rh / dh - 0.5))
        const y0 = Math.floor(fy)
        const y1 = Math.min(rh - 1, y0 + 1)
        const wy = fy - y0
        for(let x = 0; x < dw; x++){
            const fx = Math.min(rw - 1, Math.max(0, (x + 0.5) * rw / dw - 0.5))
            const x0 = Math.floor(fx)
            const x1 = Math.min(rw - 1, x0 + 1)
            const wx = fx - x0
            const a = ((ry + y0) * sw + rx + x0) * 4
            const b = ((ry + y0) * sw + rx + x1) * 4
            const c = ((ry + y1) * sw + rx + x0) * 4
            const d = ((ry + y1) * sw + rx + x1) * 4
            for(let k = 0; k < 3; k++){
                const channel = 2 - k
                const value = (rgba[a + channel] * (1 - wx) + rgba[b + channel] * wx) * (1 - wy) + (rgba[c + channel] * (1 - wx) + rgba[d + channel] * wx) * wy
                out[k * plane + y * dw + x] = (value / 255 - mean[k]) / std[k]
            }
        }
    }
    return out
}

interface Box { x: number; y: number; w: number; h: number }

async function detect(rgba: Uint8ClampedArray, width: number, height: number): Promise<Box[]> {
    const scale = Math.min(1, DET_MAX_SIDE / Math.max(width, height))
    const w = Math.max(32, Math.round(width * scale / 32) * 32)
    const h = Math.max(32, Math.round(height * scale / 32) * 32)
    const input = new ort!.Tensor('float32', toPlanes(rgba, width, height, 0, 0, width, height, w, h, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), [1, 3, h, w])
    const out = await det.run({ [det.inputNames[0]]: input })
    const prob: Float32Array = out[det.outputNames[0]].data

    // The parts of the picture where the map is above the threshold, one box for each connected part
    const seen = new Uint8Array(w * h)
    const boxes: Box[] = []
    const stack: number[] = []
    for(let y0 = 0; y0 < h; y0++){
        for(let x0 = 0; x0 < w; x0++){
            const start = y0 * w + x0
            if(seen[start] === 1 || prob[start] <= DET_THRESHOLD){
                continue
            }
            stack.length = 0
            stack.push(start)
            seen[start] = 1
            let minX = x0, maxX = x0, minY = y0, maxY = y0, sum = 0, count = 0
            while(stack.length > 0){
                const p = stack.pop() as number
                const x = p % w
                const y = (p - x) / w
                sum += prob[p]
                count++
                if(x < minX) minX = x
                if(x > maxX) maxX = x
                if(y < minY) minY = y
                if(y > maxY) maxY = y
                if(x > 0 && seen[p - 1] === 0 && prob[p - 1] > DET_THRESHOLD){ seen[p - 1] = 1; stack.push(p - 1) }
                if(x < w - 1 && seen[p + 1] === 0 && prob[p + 1] > DET_THRESHOLD){ seen[p + 1] = 1; stack.push(p + 1) }
                if(y > 0 && seen[p - w] === 0 && prob[p - w] > DET_THRESHOLD){ seen[p - w] = 1; stack.push(p - w) }
                if(y < h - 1 && seen[p + w] === 0 && prob[p + w] > DET_THRESHOLD){ seen[p + w] = 1; stack.push(p + w) }
            }
            const bw = maxX - minX + 1
            const bh = maxY - minY + 1
            if(count < 12 || Math.min(bw, bh) < 3 || sum / count < DET_BOX_SCORE){
                continue
            }
            const grow = (bw * bh * DET_UNCLIP) / (2 * (bw + bh))
            boxes.push({ x: (minX - grow) / w * width, y: (minY - grow) / h * height, w: (bw + 2 * grow) / w * width, h: (bh + 2 * grow) / h * height })
        }
    }
    return boxes
}

async function recognizeBox(rgba: Uint8ClampedArray, width: number, height: number, box: Box): Promise<{ text: string; confidence: number } | null> {
    const x = Math.max(0, Math.floor(box.x))
    const y = Math.max(0, Math.floor(box.y))
    const w = Math.min(width - x, Math.ceil(box.w))
    const h = Math.min(height - y, Math.ceil(box.h))
    if(w < 4 || h < 4){
        return null
    }
    const tw = Math.min(REC_MAX_WIDTH, Math.max(16, Math.ceil(REC_HEIGHT * w / h)))
    const input = new ort!.Tensor('float32', toPlanes(rgba, width, height, x, y, w, h, tw, REC_HEIGHT, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), [1, 3, REC_HEIGHT, tw])
    const out = await rec.run({ [rec.inputNames[0]]: input })
    const result = out[rec.outputNames[0]]
    const steps: number = result.dims[1]
    const classes: number = result.dims[2]
    const data: Float32Array = result.data
    let text = ''
    let last = 0
    let sum = 0
    let count = 0
    const letters: Array<{ at: number; text: string }> = []
    for(let t = 0; t < steps; t++){
        let best = 0
        let value = -1
        for(let c = 0; c < classes; c++){
            const v = data[t * classes + c]
            if(v > value){
                value = v
                best = c
            }
        }
        if(best !== 0 && best !== last){
            letters.push({ at: t, text: keys[best] ?? '' })
            sum += value
            count++
        }
        last = best
    }
    for(const letter of letters){
        text += letter.text
    }
    const joined = fixNearWords(splitJoinedWords(text))
    // A reading that is mostly not words is less sure than the recognizer thinks (the other reader may do better)
    return { text: joined, confidence: count > 0 ? (100 * sum / count) * (0.35 + 0.65 * plausibility(joined)) : 0 }
}

/** Where the text is: the boxes (pixels of the picture) of the lines of text the detector finds. */
export async function ppocrDetectBoxes(rgba: Uint8ClampedArray, width: number, height: number): Promise<Array<{ x: number; y: number; w: number; h: number }>> {
    if(!ppocrReady()){
        return []
    }
    const started = Date.now()
    if(remote !== null){
        // The picture is copied (the engine keeps using its own) and given to the other worker
        const copy = rgba.slice()
        const answer = await askRemote({ op: 'detect', w: width, h: height, buf: copy.buffer }, [copy.buffer])
        if(answer === null){
            return []
        }
        ppocrStats.detects++
        ppocrStats.detectMs += answer.ms
        return answer.boxes
    }
    const boxes = await detect(rgba, width, height)
    ppocrStats.detects++
    ppocrStats.detectMs += Date.now() - started
    return boxes
}

/** Reads the lines of text of a picture (a canvas): the lines come in the order of reading, with their boxes in pixels of the picture. */
export async function ppocrRecognize(canvas: any): Promise<OcrResult> {
    const width: number = canvas.width
    const height: number = canvas.height
    const context = canvas.getContext('2d')
    if(width < 8 || height < 8 || context === null){
        return { text: '', confidence: 0, lines: [] }
    }
    const rgba: Uint8ClampedArray = context.getImageData(0, 0, width, height).data
    if(remote !== null){
        const answer = await askRemote({ op: 'read', w: width, h: height, buf: rgba.buffer }, [rgba.buffer])
        if(answer === null){
            return { text: '', confidence: 0, lines: [] }
        }
        ppocrStats.reads++
        ppocrStats.ms += answer.ms
        return answer.result
    }
    const started = Date.now()
    const result = await recognizeLocal(rgba, width, height)
    ppocrStats.reads++
    ppocrStats.ms += Date.now() - started
    return result
}

/** The reading itself, with the models of this thread. */
async function recognizeLocal(rgba: Uint8ClampedArray, width: number, height: number): Promise<OcrResult> {
    const boxes = (await detect(rgba, width, height)).sort((a, b) => (a.y + a.h / 2) - (b.y + b.h / 2) || a.x - b.x)
    const lines: OcrLine[] = []
    for(const box of boxes){
        // The recognizer runs in the thread of this worker: between two lines it may do something else (answer, look at the picture)
        await new Promise<void>(resolve => setTimeout(resolve, 0))
        const read = await recognizeBox(rgba, width, height, box)
        if(read === null || read.text.trim() === ''){
            continue
        }
        lines.push({
            text: read.text.trim(),
            confidence: read.confidence,
            bbox: { x0: box.x, y0: box.y, x1: box.x + box.w, y1: box.y + box.h },
            wordHeight: box.h,
        })
    }
    if(lines.length === 0){
        return { text: '', confidence: 0, lines: [] }
    }
    return { text: lines.map(line => line.text).join('\n'), confidence: lines.reduce((sum, line) => sum + line.confidence, 0) / lines.length, lines }
}
