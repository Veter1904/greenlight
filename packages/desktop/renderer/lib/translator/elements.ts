import TranslatorEngine, { EngineStatus, Polarity, TextColor, TranslatedSegment, TranslateFunction, TranslatorResult, isSameReading, normalizeText, templateKey, toTranslatable } from './engine'
import { getReaderMode } from './ocr'
import { ppocrDetectBoxes, ppocrReady } from './ppocr'
import { Region, FitMode, regionToSource, videoRectToScreen } from './geometry'
import { getOcrWorker } from './ocr'
import { BlockRole, analyseScreen, classify } from './scout'
import { inkMasksInSlices } from './ink'
import { glossaryTranslation } from './uiwords'
import type { AutoFrame } from './autowindows'

/**
 * The translator that works with elements of the interface instead of windows and modes.
 *
 * The scout looks at the whole picture every now and then and finds the pieces of text (a tab, a button, a line of a table,
 * a subtitle). Every piece is an ELEMENT with a memory of its own:
 *  - it has a number and a place, and stays the same element from one look to the next (a small shift of the place, or
 *    a look that misses it now and then, does not make a new one);
 *  - its text pixels are remembered when it is read: as long as they stay the same, it is not read again and its
 *    translation does not change (nothing blinks); when they change, only this element is read again;
 *  - a place where nothing readable was found is remembered as a false alarm and left alone for a while;
 *  - the translation is drawn where the text is.
 * A menu is just a screen with many elements; there are no modes.
 */

export interface ElementsOptions {
    getVideo: () => HTMLVideoElement | null;
    getBox: () => { width: number; height: number };
    getSettings: () => { fit: FitMode; source: string; target: string; color: TextColor; panels?: boolean };
    // The windows the player has placed himself: those places are left to them
    getManualRegions: () => Region[];
    translate: TranslateFunction;
    // The translations that are on screen, one result per element (the role tells a subtitle from the rest)
    onResults: (results: ElementResult[]) => void;
    onFrames?: (frames: AutoFrame[]) => void;
    onStatus?: (status: EngineStatus) => void;
    // The part of the time the reader may work (0.25 = a quarter): after a reading the engine waits so that it does not take more.
    // Without it (or 1) the reader works as long as there is something to read.
    readDuty?: () => number;
    // How fast the engine reacts: 1 = as it was made for a video (a look every 0.7 s, a shown text read again after 1.8 s); less is faster
    // (0.5 = twice as fast). A device that sends only a few pictures a second, and wants new text on screen quickly, asks for less.
    speed?: () => number;
}

export interface ElementResult {
    id: number;
    role: BlockRole;
    result: TranslatorResult;
}

// How often the picture is looked at, and in what size (small text is lost in a smaller picture)
const SCAN_MS = 700
// The picture is looked at in the size it comes (up to SCAN_WIDTH): thin or faint text on a busy picture (the description of
// an item) is lost when it is made smaller. When a look takes too long it is made smaller, and bigger again when it is fast.
const SCAN_WIDTH = 1920
const SCAN_WIDTH_SMALL = 1280
const SCAN_SLOW_MS = 260
const SCAN_FAST_MS = 70
// A look in the worker that has not come back in this time is given up (it is made on the page then)
const WORKER_TIMEOUT_MS = 4000
const MAX_BLOCKS = 30
const MAX_ELEMENTS = 32

// An element that was not found on this many looks in a row is gone (a place of text that proved to be a false alarm is
// remembered for longer, so that it is not tried again and again)
const DROP_MISSES = 3
const DEAD_DROP_MISSES = 12
// How much the text pixels of an element differ from what was read (0 = the same, 1 = nothing in common): more than
// CHANGED and the element is read again; when less than GONE of the ink that was read is left, the text is gone
const CHANGED = 0.3
const GONE = 0.2
// A picture of the game that was taken another way (see verifyAgainst) differs a little from the live one even where nothing changed
const VERIFY_CHANGED = 0.68
// An element that had nothing to read stays "dead" for this long, unless its pixels change a lot
const DEAD_MS = 20000
const REVIVE_CHANGE = 0.55
// The text under a plate is watched between the looks, in a small grey copy of the picture: when it changes, the next look is
// called at once instead of waiting for its turn (a plate over a text that has gone must not hang for a second or two)
const WATCH_MS = 150
const WATCH_WIDTH = 640
const WATCH_CELLS_X = 24
const WATCH_CELLS_Y = 6
const WATCH_DIFF = 7
const WATCH_WAKE_GAP_MS = 500
// A look that finds this much changed under a plate takes the translation away at once (the new text is read after that)
const HIDE_CHANGED = 0.6
// A text that is still being typed (or scrolls) is not read: it is read when it has stayed the same between two looks. The looks come
// faster while something changes. A text that has not settled for this long is read anyway (a text that moves all the time).
const STILL_DIFF = 0.12
const FAST_LOOK_MS = 220
const SETTLE_LIMIT_MS = 5000
// Only a place of text this small (fraction of the picture) can be a text that is typed: a box with a hand-drawn frame that boils is not
const TYPING_MAX_AREA = 0.04
// A shown text whose new readings fail is kept for this many failures in a row (a reading fails now and then; a text that has really gone is seen by its pixels)
const KEEP_FAILURES = 3
// The detector of the modern reader finds the places of text the scout misses (stylized letters, text on a busy picture). It takes some
// time, so it looks at the picture when it has changed, and at least this often
const DETECT_GAP_MS = 2500
const DETECT_MAX_GAP_MS = 12000
// A busy moving picture (a screen of digits) makes the detector see letters everywhere: only lines of text are taken (wider than tall), the biggest first, and
// the detector is asked less often when it is slow
const DETECT_MIN_ASPECT = 1.0
const DETECT_MAX_BOXES = 24
const DETECT_SLOW_FACTOR = 10
const DETECT_CHANGE = 6
// Lines of text that stand one under the other with the same left edge, the same height and a small gap are one paragraph: they are read and translated together
const PARAGRAPH_GAP = 0.7
const PARAGRAPH_MAX_LINES = 14
// An element is not read more often than this (a text that is still the same is not worth it)
const READ_GAP_MS = 600
const SHOWN_READ_GAP_MS = 1800
// A shift of the place smaller than this is not followed (the picture is looked at coarsely and the place shakes a little)
const MOVE_TOLERANCE = 0.006

// A text that scrolls: a piece of text that was not found where an element was, but looks just like that element (the pixels of the letters are the
// same) somewhere not far from it, is that element: it has moved. Its translation goes with it and nothing is read again.
// ... the coarse look at the letters must say they are alike (this only picks the candidates), then the text pixels themselves must lie on each other
const SCROLL_COARSE = 0.42
const SCROLL_PATCH = 0.5
// A line that is cut by the edge of its window: the visible part is between these shares of the whole line, and the letters in it must lie on the same part of the whole
const CLIP_MIN = 0.25
const CLIP_MAX = 0.8
const CLIP_PATCH = 0.55
// The fastest a text may scroll (screens per second) for the speed to be believed
const MOTION_MAX = 1.5
// A piece of text that is much smaller than the lines next to it is the end of a line that is cut by an edge: it waits longer (more looks) before it is read
const SLIVER_SHARE = 0.6
const SLIVER_LOOKS = 3
// ... how far it may have gone (fractions of the screen)
const SCROLL_MAX_DX = 0.04
const SCROLL_MAX_DY = 0.4
// The place of an element whose pixels differ more than this from the text it was read with is looked at for another element that matches better
const PLACE_MISMATCH = 0.45
// Elements that left the screen are remembered (with their translations) so that a text that scrolls back is shown at once
const PARKED_MAX = 80
const PARKED_MAX_AGE_MS = 120000
// How many texts were followed when they scrolled, were put back from the memory, and were put into it (for tests)
export const scrollStats = { moved: 0, restored: 0, parked: 0, clipped: 0, scores: [] as number[] }
// A translation that has got a better version is replaced only while it is young
const BETTER_MS = 4000
// Pieces of text that are at least this tall (fractions of the screen) and not further apart than this many heights are one title
const TITLE_MIN_HEIGHT = 0.03
const TITLE_GAP = 0.7
// More ink than this between two pieces (an icon) and they are not one title
const GAP_INK = 0.3
// ... or when a line of ink runs through the gap: a column that is this much ink
const GAP_LINE = 0.75
// ... and one row of text (a block of several rows or a picture is not a title)
const TITLE_MAX_HEIGHT = 0.08
// Up to this many letters is a short text: it has to be read with at least this confidence, or it is not believed
const SHORT_TEXT_LETTERS = 6
const SHORT_TEXT_CONFIDENCE = 68
// A reading that the reader is this unsure about is not believed at all (pictures and noise)
const MIN_CONFIDENCE = 55
// A reading this sure is taken at once; a weaker one is tried again in another way, and the best one is taken
const GOOD_CONFIDENCE = 72
// ... and covers at least this part of the place (the rest of a paragraph is tried in other ways)
const GOOD_COVERAGE = 0.6
// The picture in grey is tried (it takes time) only for places that look like text to the scout
const RAW_MIN_SCORE = 50

const LETTER = new RegExp('\\p{L}', 'gu')

interface Segment extends TranslatedSegment {
    template: string;
    shownAt: number;
    // The colour of the interface behind the text as it was measured (the plate colours are made from it and the colour of the letters)
    base?: [number, number, number];
    // The colour of the letters now
    ink?: [number, number, number];
    // When the colour of the background was last followed, the picture behind the text at the first look and the colour of the plate that belonged to it
    baseAt?: number;
    behindRef?: [number, number, number];
    base0?: [number, number, number];
    // Brightness of the letters as the reader saw them (the first colour of the plate): the colour of the ink must be on the same side of the background
    fgLuma0?: number;
}

interface Element {
    id: number;
    // Where it is (fractions of the stream area) and where its text pixels are in the picture that was looked at (fractions)
    rect: Region;
    scan: Region;
    polarity: Polarity;
    role: BlockRole;
    plate: boolean;
    panel?: Region;
    // How much the place looked like text to the scout
    score: number;
    // On how many looks in a row it was found / not found
    seen: number;
    missed: number;
    // new: not read yet; text: has a translation; dead: nothing readable (a false alarm); idle: its text has gone
    state: 'new' | 'text' | 'dead' | 'idle';
    dirty: boolean;
    // A small grey fingerprint of the place of the text while it is shown (the watch between the looks), and the reading it belongs to
    detected?: boolean;
    // The pixels of the text changed much since it was read: a reading that finds nothing now takes the plate away at once
    strongChange?: boolean;
    watchSig?: Uint8Array;
    // The signature of the last look and whether the text changed since the look before (typing); since when it has not settled
    lastCells?: Uint8Array;
    changing?: boolean;
    changingSince?: number;
    watchFor?: number;
    dirtySince: number;
    reading: boolean;
    readAt: number;
    deadUntil: number;
    deadCount: number;
    // The text pixels when it was read (cells), and how much ink there was
    sig: Uint8Array | null;
    sigSum: number;
    sigGrid?: { gx: number; gy: number };
    // The text pixels as they were when it was read (to know the text again when it has moved)
    sigPatch?: Patch;
    // When the place was last confirmed, how fast the text is moving, and which part of it is seen when the edge of a window cuts it
    placedAt?: number;
    motion?: { vx: number; vy: number; at: number; sustained: boolean };
    clip?: Region;
    segments: Segment[];
}

interface Cells {
    cells: Uint8Array;
    sum: number;
    // The size of the grid of cells (a picture of another size is compared on the same grid)
    gx: number;
    gy: number;
}

// How much of the first box lies inside the second one (0..1)
/** "#rrggbb" or "rgb(r, g, b)" as numbers; null when it is neither. */
function cssColor(value: string | undefined): number[] | null {
    if(value === undefined){
        return null
    }
    const hex = /^#([0-9a-f]{6})$/i.exec(value)
    if(hex !== null){
        return [parseInt(hex[1].slice(0, 2), 16), parseInt(hex[1].slice(2, 4), 16), parseInt(hex[1].slice(4, 6), 16)]
    }
    const rgb = /^rgba?\((\d+),\s*(\d+),\s*(\d+)/.exec(value)
    return rgb === null ? null : [Number(rgb[1]), Number(rgb[2]), Number(rgb[3])]
}

function inside(inner: Region, outer: Region): number {
    const width = Math.min(inner.x + inner.w, outer.x + outer.w) - Math.max(inner.x, outer.x)
    const height = Math.min(inner.y + inner.h, outer.y + outer.h) - Math.max(inner.y, outer.y)
    if(width <= 0 || height <= 0){
        return 0
    }
    return (width * height) / Math.max(1e-9, inner.w * inner.h)
}

function center(region: Region): { x: number; y: number } {
    return { x: region.x + region.w / 2, y: region.y + region.h / 2 }
}

function distance(a: Region, b: Region): number {
    const first = center(a)
    const second = center(b)
    return Math.hypot(first.x - second.x, first.y - second.y)
}

function similarSize(a: Region, b: Region): boolean {
    return a.w > b.w * 0.75 && a.w < b.w * 1.33 && a.h > b.h * 0.65 && a.h < b.h * 1.5
}

function centralness(region: Region): number {
    const middle = center(region)
    return Math.max(0, 1 - Math.hypot((middle.x - 0.5) / 0.35, (middle.y - 0.5) / 0.35))
}

// How far the colour of the letters has to move before the plate takes the new colour
const INK_CHANGED = 60
// The letters have to differ at least this much in brightness from the plate, or the plate is made darker / lighter
const INK_CONTRAST = 110

function brightnessOf(color: number[]): number {
    return (color[0] * 299 + color[1] * 587 + color[2] * 114) / 1000
}

// A pixel is "coloured" when its brightest and darkest channel are this far apart (the yellow of a letter, not the grey of a picture)
const CHROMA = 90
// The mask of dark letters is an outline (the letters are inside it) when it is this dark
const OUTLINE_BLACK = 45

/**
 * The colour of the letters in a place of the picture, or null when there is not enough to tell. Letters that are drawn with
 * a dark outline (a lot of games) are found by their outline, so the mask alone would give black: when the place has coloured
 * pixels, their mean is the colour of the letters; otherwise it is the mean of the text mask.
 */
function inkColor(pixels: Uint8ClampedArray, mask: Uint8Array, width: number, height: number, box: Region, polarity: Polarity): [number, number, number] | null {
    const x0 = Math.max(0, Math.floor(box.x * width))
    const y0 = Math.max(0, Math.floor(box.y * height))
    const x1 = Math.min(width, Math.ceil((box.x + box.w) * width))
    const y1 = Math.min(height, Math.ceil((box.y + box.h) * height))
    let r = 0
    let g = 0
    let b = 0
    let count = 0
    let cr = 0
    let cg = 0
    let cb = 0
    let colored = 0
    for(let y = y0; y < y1; y++){
        for(let x = x0; x < x1; x++){
            const i = y * width + x
            const pr = pixels[i * 4]
            const pg = pixels[i * 4 + 1]
            const pb = pixels[i * 4 + 2]
            if(Math.max(pr, pg, pb) - Math.min(pr, pg, pb) > CHROMA){
                cr += pr
                cg += pg
                cb += pb
                colored++
            }
            if(mask[i] === 1){
                r += pr
                g += pg
                b += pb
                count++
            }
        }
    }
    if(count < 8){
        return colored >= 12 ? [Math.round(cr / colored), Math.round(cg / colored), Math.round(cb / colored)] : null
    }
    const mean: [number, number, number] = [Math.round(r / count), Math.round(g / count), Math.round(b / count)]
    // Dark letters that are really a dark outline (the mask is nearly black) have their colour inside; dark letters that are not black
    // (dark teal on a bright button) are what they are, the coloured pixels there are the button
    const outline = polarity === 'dark' && brightnessOf(mean) < OUTLINE_BLACK
    if((polarity === 'light' || outline) && colored >= 12 && colored >= count * 0.15){
        return [Math.round(cr / colored), Math.round(cg / colored), Math.round(cb / colored)]
    }
    return mean
}

/** The colours of a plate: the text keeps the colour of the original letters; the plate is made dark or light enough to read it. */
/** The mean colour of the pixels of a place that are not letters (what the plate imitates). */
function backgroundColor(pixels: Uint8ClampedArray, mask: Uint8Array, width: number, height: number, box: Region): [number, number, number] | null {
    const x0 = Math.max(0, Math.floor(box.x * width))
    const y0 = Math.max(0, Math.floor(box.y * height))
    const x1 = Math.min(width, Math.ceil((box.x + box.w) * width))
    const y1 = Math.min(height, Math.ceil((box.y + box.h) * height))
    let r = 0
    let g = 0
    let b = 0
    let count = 0
    for(let y = y0; y < y1; y += 2){
        for(let x = x0; x < x1; x += 2){
            const i = y * width + x
            if(mask[i] === 0){
                r += pixels[i * 4]
                g += pixels[i * 4 + 1]
                b += pixels[i * 4 + 2]
                count++
            }
        }
    }
    return count < 12 ? null : [Math.round(r / count), Math.round(g / count), Math.round(b / count)]
}

// The plate follows the colour of the picture behind the text (a moving background, a menu entry that is chosen) without a new reading: the letters
// are what the engine holds on to, the background is only imitated
const BACKGROUND_CHANGED = 24
const BACKGROUND_EVERY_MS = 700

function plateColors(base: [number, number, number], ink: [number, number, number]): { background: [number, number, number]; foreground: string } {
    const baseLight = brightnessOf(base)
    const inkLight = brightnessOf(ink)
    let background = base
    if(Math.abs(inkLight - baseLight) < INK_CONTRAST){
        background = inkLight >= baseLight
            ? [Math.round(base[0] * 0.25), Math.round(base[1] * 0.25), Math.round(base[2] * 0.25)]
            : [Math.round(base[0] + (255 - base[0]) * 0.75), Math.round(base[1] + (255 - base[1]) * 0.75), Math.round(base[2] + (255 - base[2]) * 0.75)]
    }
    // Still too close (dark letters on a dark picture): black or white text instead
    if(Math.abs(inkLight - brightnessOf(background)) < 70){
        return { background, foreground: brightnessOf(background) > 150 ? '#111111' : '#ffffff' }
    }
    return { background, foreground: 'rgb(' + ink[0] + ', ' + ink[1] + ', ' + ink[2] + ')' }
}

/** The text pixels of a place as they are: where the letters are (positions in the box, row by row). */
interface Patch {
    w: number;
    h: number;
    ink: Int32Array;
    // The rows and columns the letters really take (inclusive): the box around them is not exact
    top: number;
    bottom: number;
    left: number;
    right: number;
}

/** The rows and columns the letters take at a place now (rows with a few ink pixels only count: a speck is not a letter). */
function inkExtent(mask: Uint8Array, width: number, height: number, box: Region): { top: number; bottom: number; left: number; right: number } | null {
    const x0 = Math.max(0, Math.floor(box.x * width))
    const y0 = Math.max(0, Math.floor(box.y * height))
    const x1 = Math.min(width, Math.ceil((box.x + box.w) * width))
    const y1 = Math.min(height, Math.ceil((box.y + box.h) * height))
    let top = -1
    let bottom = -1
    let left = Infinity
    let right = -1
    for(let y = y0; y < y1; y++){
        let count = 0
        let first = -1
        let last = -1
        for(let x = x0; x < x1; x++){
            if(mask[y * width + x] === 1){
                count++
                if(first < 0){
                    first = x
                }
                last = x
            }
        }
        if(count >= 2){
            if(top < 0){
                top = y
            }
            bottom = y
            left = Math.min(left, first)
            right = Math.max(right, last)
        }
    }
    return top < 0 ? null : { top, bottom, left, right }
}

function patchOf(mask: Uint8Array, width: number, height: number, box: Region): Patch | null {
    const x0 = Math.max(0, Math.floor(box.x * width))
    const y0 = Math.max(0, Math.floor(box.y * height))
    const x1 = Math.min(width, Math.ceil((box.x + box.w) * width))
    const y1 = Math.min(height, Math.ceil((box.y + box.h) * height))
    if(x1 <= x0 || y1 <= y0 || (x1 - x0) * (y1 - y0) > 400000){
        return null
    }
    const w = x1 - x0
    const list: number[] = []
    for(let y = y0; y < y1; y++){
        for(let x = x0; x < x1; x++){
            if(mask[y * width + x] === 1){
                list.push((y - y0) * w + (x - x0))
            }
        }
    }
    if(list.length < 20){
        return null
    }
    let top = Infinity
    let bottom = -1
    let left = Infinity
    let right = -1
    for(const position of list){
        const row = Math.floor(position / w)
        const column = position % w
        top = Math.min(top, row)
        bottom = Math.max(bottom, row)
        left = Math.min(left, column)
        right = Math.max(right, column)
    }
    return { w, h: y1 - y0, ink: Int32Array.from(list), top, bottom, left, right }
}

/** How well the letters of a patch lie on the letters at a place now (0..1: the same text, a few pixels off, is about 0.7 and more; another text 0.3 and less). */
function patchMatch(patch: Patch, mask: Uint8Array, width: number, height: number, box: Region): number {
    const x0 = Math.floor(box.x * width)
    const y0 = Math.floor(box.y * height)
    const x1 = Math.min(width, Math.ceil((box.x + box.w) * width))
    const y1 = Math.min(height, Math.ceil((box.y + box.h) * height))
    let there = 0
    for(let y = Math.max(0, y0); y < y1; y++){
        for(let x = Math.max(0, x0); x < x1; x++){
            if(mask[y * width + x] === 1){
                there++
            }
        }
    }
    let best = 0
    for(let dy = -2; dy <= 2; dy++){
        for(let dx = -3; dx <= 3; dx++){
            let hits = 0
            for(let i = 0; i < patch.ink.length; i++){
                const x = x0 + dx + (patch.ink[i] % patch.w)
                const y = y0 + dy + Math.floor(patch.ink[i] / patch.w)
                if(x >= 0 && x < width && y >= 0 && y < height && mask[y * width + x] === 1){
                    hits++
                }
            }
            best = Math.max(best, 2 * hits / (patch.ink.length + there))
        }
    }
    return best
}

/**
 * A line cut by the edge of its window: only the upper (edge 'top') or the lower (edge 'bottom') part of the patch is seen in the box. How well do the
 * letters of that part lie on the letters in the box (0..1)?
 */
function patchMatchPartial(patch: Patch, mask: Uint8Array, width: number, height: number, extent: { top: number; bottom: number; left: number; right: number }, edge: 'top' | 'bottom'): number {
    const seen = extent.bottom - extent.top + 1
    const rowStart = edge === 'top' ? patch.top : patch.bottom - seen + 1
    let used = 0
    for(let i = 0; i < patch.ink.length; i++){
        const row = Math.floor(patch.ink[i] / patch.w)
        if(row >= rowStart && row < rowStart + seen){
            used++
        }
    }
    if(used < 12){
        return 0
    }
    let there = 0
    for(let y = extent.top; y <= extent.bottom; y++){
        for(let x = extent.left; x <= extent.right; x++){
            if(mask[y * width + x] === 1){
                there++
            }
        }
    }
    let best = 0
    for(let dy = -1; dy <= 1; dy++){
        for(let dx = -3; dx <= 3; dx++){
            let hits = 0
            for(let i = 0; i < patch.ink.length; i++){
                const row = Math.floor(patch.ink[i] / patch.w)
                if(row < rowStart || row >= rowStart + seen){
                    continue
                }
                const x = extent.left + dx + (patch.ink[i] % patch.w - patch.left)
                const y = extent.top + dy + (row - rowStart)
                if(x >= 0 && x < width && y >= 0 && y < height && mask[y * width + x] === 1){
                    hits++
                }
            }
            best = Math.max(best, 2 * hits / (used + there))
        }
    }
    return best
}

/** The text pixels of a place as a coarse grid: how much of every cell is text (0..255). */
function cellsOf(mask: Uint8Array, width: number, height: number, box: Region, grid?: { gx: number; gy: number }): Cells {
    const x0 = Math.max(0, Math.floor(box.x * width))
    const y0 = Math.max(0, Math.floor(box.y * height))
    const x1 = Math.min(width, Math.ceil((box.x + box.w) * width))
    const y1 = Math.min(height, Math.ceil((box.y + box.h) * height))
    const gx = grid === undefined ? Math.max(6, Math.min(64, Math.round((x1 - x0) / 3))) : grid.gx
    const gy = grid === undefined ? Math.max(3, Math.min(24, Math.round((y1 - y0) / 3))) : grid.gy
    const cells = new Uint8Array(gx * gy)
    let sum = 0

    if(x1 <= x0 || y1 <= y0){
        return { cells, sum, gx, gy }
    }

    const counts = new Uint16Array(gx * gy)
    const totals = new Uint16Array(gx * gy)
    for(let y = y0; y < y1; y++){
        const cy = Math.min(gy - 1, Math.floor((y - y0) * gy / (y1 - y0)))
        for(let x = x0; x < x1; x++){
            const cell = cy * gx + Math.min(gx - 1, Math.floor((x - x0) * gx / (x1 - x0)))
            totals[cell]++
            if(mask[y * width + x] === 1){
                counts[cell]++
            }
        }
    }
    for(let i = 0; i < cells.length; i++){
        cells[i] = totals[i] === 0 ? 0 : Math.round(255 * counts[i] / totals[i])
        sum += cells[i]
    }
    return { cells, sum, gx, gy }
}

/** How different two grids are: 0 the same, 1 nothing in common. */
function cellsDifference(a: Uint8Array, b: Uint8Array): number {
    if(a.length !== b.length){
        return 1
    }
    let different = 0
    let union = 0
    for(let i = 0; i < a.length; i++){
        different += Math.abs(a[i] - b[i])
        union += Math.max(a[i], b[i])
    }
    return union === 0 ? 0 : different / union
}

/**
 * How different two signatures are, looking only at the cells that hold much ink in one of them: the letters. A text on a transparent background (a
 * moving picture behind it) gets ink in many cells by chance, which is not a change of the text; the cells of the strokes are what counts.
 */
const STRONG_CELL = 110
function coreDifference(a: Uint8Array, b: Uint8Array): number {
    if(a.length !== b.length){
        return 1
    }
    let different = 0
    let union = 0
    for(let i = 0; i < a.length; i++){
        if(Math.max(a[i], b[i]) >= STRONG_CELL){
            different += Math.abs(a[i] - b[i])
            union += Math.max(a[i], b[i])
        }
    }
    return union === 0 ? cellsDifference(a, b) : different / union
}

export default class ElementsEngine {
    _options: ElementsOptions
    _running = false
    _ready = false
    _timer: ReturnType<typeof setTimeout> | undefined
    _watchTimer: ReturnType<typeof setTimeout> | undefined
    _watchCanvas: HTMLCanvasElement | null = null
    _scanning = false
    // How many looks were made (for the performance report)
    _scans = 0
    _detected: Array<{ x: number; y: number; w: number; h: number; polarity: Polarity }> = []
    _detectedAt = 0
    _detectMs = 0
    _lastDetectMs = 0
    _detectThumb: Uint8Array | null = null
    _lookChanged = false
    _lastWake = 0
    _readTimer: ReturnType<typeof setTimeout> | undefined
    _elements: Element[] = []
    _nextId = 1
    _canvas = document.createElement('canvas')
    _scanTook = 0
    _width = SCAN_WIDTH
    // The text pixels of the last look (for the signatures of the elements)
    _masks: Array<{ mask: Uint8Array; count: number }> = []
    _scanSize = { width: 0, height: 0 }
    _polarities: Polarity[] = ['light', 'dark']
    // The tools of the text reader and the translations that were made
    _toolbox: TranslatorEngine
    _cache: Map<string, string> = new Map()
    // Elements that have gone out of sight (scrolled away) with what was read: they come back when their text does
    _parked: Array<{ element: Element; at: number }> = []
    _pending: Map<string, Promise<string>> = new Map()
    _lastKey = ''
    // The worker that does the analysis of the picture (null: it is done here), and the looks that wait for it
    _worker: Worker | null = null
    _requestId = 0
    _pendingLooks: Map<number, (answer: any) => void> = new Map()
    // What the last look found (for tests)
    _lastPieces: Array<{ x: number; y: number; w: number; h: number; polarity: Polarity; role: BlockRole; score: number }> = []

    constructor(options: ElementsOptions) {
        this._options = options
        // The reader of the translator engine does the reading of one place (it is not started: it is only a toolbox)
        this._toolbox = new TranslatorEngine({
            getVideo: options.getVideo,
            getBox: options.getBox,
            getConfig: () => ({ region: { x: 0, y: 0, w: 1, h: 1 }, color: 'auto', source: options.getSettings().source, target: options.getSettings().target, fit: options.getSettings().fit }),
            translate: options.translate,
            onResult: () => { /* not used */ },
            onStatus: () => { /* not used */ },
            priority: 1,
        })
    }

    /**
     * Looks at the picture: in the worker when there is one (the page does not stop for it), otherwise here. The pixels go to
     * the worker without being copied and the text pixels come back the same way.
     */
    /** The boxes of lines that make a paragraph (same left edge, same height, a small gap between) become one box. */
    _joinParagraphs(boxes: Array<{ x: number; y: number; w: number; h: number }>): Array<{ x: number; y: number; w: number; h: number }> {
        const sorted = boxes.slice().sort((a, b) => a.y - b.y || a.x - b.x)
        const groups: Array<{ x: number; y: number; w: number; h: number; lines: number; lastH: number; lastX: number }> = []
        for(const box of sorted){
            const group = groups.find((g) => {
                if(g.lines >= PARAGRAPH_MAX_LINES){
                    return false
                }
                const bottom = g.y + g.h
                const gap = box.y - bottom
                const similar = box.h > g.lastH * 0.6 && box.h < g.lastH * 1.6
                const sameEdge = Math.abs(box.x - g.lastX) <= Math.max(g.lastH, box.h) * 1.5 || (box.x < g.x + g.w && box.x + box.w > g.x && Math.abs(box.x - g.x) <= Math.max(g.lastH, box.h) * 4)
                return similar && sameEdge && gap >= -box.h * 0.3 && gap <= Math.max(g.lastH, box.h) * PARAGRAPH_GAP
            })
            if(group === undefined){
                groups.push({ ...box, lines: 1, lastH: box.h, lastX: box.x })
            } else {
                const right = Math.max(group.x + group.w, box.x + box.w)
                const bottom = Math.max(group.y + group.h, box.y + box.h)
                group.x = Math.min(group.x, box.x)
                group.y = Math.min(group.y, box.y)
                group.w = right - group.x
                group.h = bottom - group.y
                group.lines++
                group.lastH = box.h
                group.lastX = box.x
            }
        }
        return groups.map((g) => ({ x: g.x, y: g.y, w: g.w, h: g.h }))
    }

    /** The places of text the detector finds (fractions of the picture), taken again when the picture has changed. */
    async _detectBlocks(pixels: Uint8ClampedArray, width: number, height: number): Promise<Array<{ x: number; y: number; w: number; h: number; polarity: Polarity }>> {
        const now = Date.now()
        // A coarse grey copy of the picture: has it changed since the detector looked?
        const tw = 64
        const th = 36
        const thumb = new Uint8Array(tw * th)
        for(let y = 0; y < th; y++){
            for(let x = 0; x < tw; x++){
                const o = ((Math.floor((y + 0.5) * height / th)) * width + Math.floor((x + 0.5) * width / tw)) * 4
                thumb[y * tw + x] = (pixels[o] * 299 + pixels[o + 1] * 587 + pixels[o + 2] * 114) / 1000
            }
        }
        let change = 0
        if(this._detectThumb !== null){
            for(let i = 0; i < thumb.length; i++){
                change += Math.abs(thumb[i] - this._detectThumb[i])
            }
            change /= thumb.length
        }
        const due = this._detectThumb === null || now - this._detectedAt >= DETECT_MAX_GAP_MS || (change > DETECT_CHANGE && now - this._detectedAt >= Math.max(DETECT_GAP_MS, this._lastDetectMs * DETECT_SLOW_FACTOR))
        if(!due){
            return this._detected
        }
        this._detectThumb = thumb
        this._detectedAt = now
        const detectStarted = Date.now()
        const boxes = await ppocrDetectBoxes(pixels, width, height)
        this._lastDetectMs = Date.now() - detectStarted
        this._detectMs += this._lastDetectMs
        const lines = this._joinParagraphs(boxes).filter(box => box.w >= box.h * DETECT_MIN_ASPECT).sort((a, b) => b.w * b.h - a.w * a.h).slice(0, DETECT_MAX_BOXES)
        this._detected = lines.map((box) => {
            // The letters are the smaller part of the box: bright ones on a dark ground, or dark ones on a light ground
            const x0 = Math.max(0, Math.floor(box.x))
            const y0 = Math.max(0, Math.floor(box.y))
            const x1 = Math.min(width, Math.ceil(box.x + box.w))
            const y1 = Math.min(height, Math.ceil(box.y + box.h))
            let low = 255
            let high = 0
            for(let y = y0; y < y1; y += 2){
                for(let x = x0; x < x1; x += 2){
                    const o = (y * width + x) * 4
                    const grey = (pixels[o] * 299 + pixels[o + 1] * 587 + pixels[o + 2] * 114) / 1000
                    low = Math.min(low, grey)
                    high = Math.max(high, grey)
                }
            }
            const middle = (low + high) / 2
            let bright = 0
            let total = 0
            for(let y = y0; y < y1; y += 2){
                for(let x = x0; x < x1; x += 2){
                    const o = (y * width + x) * 4
                    if((pixels[o] * 299 + pixels[o + 1] * 587 + pixels[o + 2] * 114) / 1000 > middle){
                        bright++
                    }
                    total++
                }
            }
            return { x: box.x / width, y: box.y / height, w: box.w / width, h: box.h / height, polarity: (bright < total * 0.5 ? 'light' : 'dark') as Polarity }
        })
        return this._detected
    }

    _analyse(pixels: Uint8ClampedArray, width: number, height: number): Promise<{ blocks: ReturnType<typeof analyseScreen>['blocks']; masks: Array<{ mask: Uint8Array; count: number }> }> {
        const worker = this._worker
        if(worker === null){
            return Promise.resolve(analyseScreen(pixels, width, height, this._polarities, MAX_BLOCKS))
        }

        return new Promise((resolve) => {
            const id = ++this._requestId
            const timer = setTimeout(() => {
                // The worker does not answer: it is given up (the pictures that were sent to it are gone) and the page does the looks
                this._pendingLooks.delete(id)
                this._dropWorker()
                resolve({ blocks: [], masks: [] })
            }, WORKER_TIMEOUT_MS)
            this._pendingLooks.set(id, (answer: any) => {
                clearTimeout(timer)
                resolve(answer)
            })

            try {
                worker.postMessage({ id, buffer: pixels.buffer, width, height, polarities: this._polarities, limit: MAX_BLOCKS }, [pixels.buffer])
            } catch(error) {
                clearTimeout(timer)
                this._pendingLooks.delete(id)
                this._dropWorker()
                resolve(analyseScreen(pixels, width, height, this._polarities, MAX_BLOCKS))
            }
        })
    }

    _dropWorker() {
        if(this._worker !== null){
            this._worker.terminate()
            this._worker = null
        }
    }

    start() {
        if(this._running){
            return
        }
        this._running = true

        // The scout runs in a worker when the page can make one
        if(typeof Worker !== 'undefined' && typeof window !== 'undefined'){
            try {
                const worker = new Worker(new URL('/scout/worker.js', window.location.href).href)
                worker.onmessage = (event: MessageEvent) => {
                    const answer = event.data
                    const waiting = this._pendingLooks.get(answer.id)
                    if(waiting !== undefined){
                        this._pendingLooks.delete(answer.id)
                        if(answer.error !== undefined){
                            // The worker could not do it: from now on the page does
                            this._dropWorker()
                            waiting({ blocks: [], masks: [] })
                            return
                        }
                        waiting(answer)
                    }
                }
                worker.onerror = () => {
                    this._dropWorker()
                }
                this._worker = worker
            } catch(error) {
                this._worker = null
            }
        }
        this._options.onStatus?.('loading')

        getOcrWorker(this._options.getSettings().source).then(() => {
            if(this._running){
                this._ready = true
                this._options.onStatus?.('ready')
            }
        }).catch((error) => {
            console.error('Translator: failed to start the text reader:', error)
            if(this._running){
                this._options.onStatus?.('error')
            }
        })

        this._timer = setTimeout(() => this._scan(), 300)
        this._watchTimer = setTimeout(() => this._watch(), 600)
        this._readTimer = setTimeout(() => this._readLoop(), 500)
    }

    stop() {
        this._running = false
        if(this._timer !== undefined){
            clearTimeout(this._timer)
            this._timer = undefined
        }
        clearTimeout(this._watchTimer)
        if(this._readTimer !== undefined){
            clearTimeout(this._readTimer)
            this._readTimer = undefined
        }
        this._dropWorker()
        this._pendingLooks.clear()
        this._elements = []
        this._lastKey = ''
        this._options.onResults([])
        this._options.onFrames?.([])
        this._cache.clear()
        this._parked = []
        this._pending.clear()
    }

    // ----------------------------------------------------------------------------------------------- looking

    async _scan() {
        if(!this._running){
            return
        }

        const began = Date.now()
        this._scanning = true
        this._scans++
        try {
            await this._scanOnce()
        } catch(error) {
            console.warn('Translator: look failed:', error)
        }
        this._scanning = false
        // The time of the detector is not the time of the look: it must not make the look smaller and bigger again (everything would seem changed)
        this._scanTook = Math.max(1, Date.now() - began - this._detectMs)
        this._detectMs = 0
        if(this._width > SCAN_WIDTH_SMALL && this._scanTook > SCAN_SLOW_MS){
            this._width = SCAN_WIDTH_SMALL
        } else if(this._width < SCAN_WIDTH && this._scanTook < SCAN_FAST_MS){
            this._width = SCAN_WIDTH
        }

        if(this._running){
            const speed = this._options.speed?.() ?? 1
            const fast = this._lookChanged
            this._lookChanged = false
            this._timer = setTimeout(() => this._scan(), fast ? Math.max(FAST_LOOK_MS, this._scanTook * 3) : Math.max(SCAN_MS * speed, this._scanTook * 8 * Math.max(0.5, speed)))
        }
    }

    /** Between the looks: is the text under a plate still there? A change calls the next look at once. */
    async _watch() {
        if(!this._running){
            return
        }
        try {
            this._watchOnce()
        } catch(error) {
            // a failed watch is only a slower reaction
        }
        if(this._running){
            this._watchTimer = setTimeout(() => this._watch(), WATCH_MS)
        }
    }

    _watchOnce() {
        const shown = this._elements.filter(element => element.state === 'text' && element.segments.length > 0 && !element.reading && !element.dirty)
        if(shown.length === 0 || document.hidden){
            return
        }
        const video = this._options.getVideo()
        if(video === null || video.readyState < 2){
            return
        }
        const width = WATCH_WIDTH
        const height = Math.max(8, Math.round(width * video.videoHeight / Math.max(1, video.videoWidth)))
        if(this._watchCanvas === null){
            this._watchCanvas = document.createElement('canvas')
        }
        this._watchCanvas.width = width
        this._watchCanvas.height = height
        const context = this._watchCanvas.getContext('2d', { willReadFrequently: true })
        if(context === null){
            return
        }
        // the whole picture of the game (the plates are not in it: it is the picture the engine reads)
        context.drawImage(video, 0, 0, video.videoWidth, video.videoHeight, 0, 0, width, height)
        const data = context.getImageData(0, 0, width, height).data
        let wake = false
        for(const element of shown){
            const sig = this._fingerprint(data, width, height, element.scan)
            if(element.watchSig === undefined || element.watchFor !== element.readAt){
                element.watchSig = sig
                element.watchFor = element.readAt
                continue
            }
            let sum = 0
            for(let i = 0; i < sig.length; i++){
                sum += Math.abs(sig[i] - element.watchSig[i])
            }
            if(sum / sig.length > WATCH_DIFF){
                element.watchSig = sig
                wake = true
            }
        }
        if(wake && !this._scanning && Date.now() - this._lastWake >= WATCH_WAKE_GAP_MS){
            this._lastWake = Date.now()
            clearTimeout(this._timer)
            this._timer = setTimeout(() => this._scan(), 10)
        }
    }

    /** Mean grey of a grid of cells over a place (fractions of the picture). */
    _fingerprint(data: Uint8ClampedArray, width: number, height: number, place: Region): Uint8Array {
        const out = new Uint8Array(WATCH_CELLS_X * WATCH_CELLS_Y)
        const x0 = Math.max(0, Math.floor(place.x * width))
        const y0 = Math.max(0, Math.floor(place.y * height))
        const w = Math.max(1, Math.min(width - x0, Math.ceil(place.w * width)))
        const h = Math.max(1, Math.min(height - y0, Math.ceil(place.h * height)))
        for(let cy = 0; cy < WATCH_CELLS_Y; cy++){
            for(let cx = 0; cx < WATCH_CELLS_X; cx++){
                const ax = x0 + Math.floor(cx * w / WATCH_CELLS_X)
                const bx = Math.max(ax + 1, x0 + Math.floor((cx + 1) * w / WATCH_CELLS_X))
                const ay = y0 + Math.floor(cy * h / WATCH_CELLS_Y)
                const by = Math.max(ay + 1, y0 + Math.floor((cy + 1) * h / WATCH_CELLS_Y))
                let sum = 0
                let count = 0
                for(let y = ay; y < by && y < height; y++){
                    for(let x = ax; x < bx && x < width; x++){
                        const o = (y * width + x) * 4
                        sum += data[o] * 0.299 + data[o + 1] * 0.587 + data[o + 2] * 0.114
                        count++
                    }
                }
                out[cy * WATCH_CELLS_X + cx] = count > 0 ? Math.round(sum / count) : 0
            }
        }
        return out
    }

    async _scanOnce() {
        if(document.hidden){
            return
        }

        const video = this._options.getVideo()
        if(video === null || video.readyState < 2){
            return
        }

        const box = this._options.getBox()
        const settings = this._options.getSettings()
        const visible = regionToSource(video.videoWidth, video.videoHeight, box.width, box.height, settings.fit, { x: 0, y: 0, w: 1, h: 1 })
        if(visible === null){
            return
        }

        const width = Math.min(this._width, Math.round(visible.sw))
        const height = Math.max(8, Math.round(width * visible.sh / visible.sw))
        this._canvas.width = width
        this._canvas.height = height
        const context = this._canvas.getContext('2d', { willReadFrequently: true })
        if(context === null){
            return
        }
        context.drawImage(video, visible.sx, visible.sy, visible.sw, visible.sh, 0, 0, width, height)
        // The copy of the frame and its reading are two tasks, not one: each is short, so the picture of the game and the pad are
        // not held up for the sum of them (on the Deck 14 ms and 8 ms, and the page waits for 22 ms in one piece)
        await new Promise<void>(resolve => setTimeout(resolve, 0))
        if(!this._running){
            return
        }
        const pixels = context.getImageData(0, 0, width, height).data

        // The detector of the modern reader (when it is loaded) adds the places of text the scout did not find
        const detected = getReaderMode() !== 'tesseract' && ppocrReady() ? await this._detectBlocks(pixels, width, height) : []
        const analysed = await this._analyse(pixels, width, height)
        const masks = analysed.masks
        const blocks = analysed.blocks.slice()
        for(const found of detected){
            if(blocks.length >= MAX_BLOCKS){
                break
            }
            const area = found.w * found.h
            const known = blocks.some(block => inside(found, block) > 0.5 || (inside(block, found) > 0.7 && block.w * block.h > area * 0.3))
            if(!known && area > 0){
                blocks.push({ x: found.x, y: found.y, w: found.w, h: found.h, polarity: found.polarity, role: classify(found), score: 300, plate: false, detected: true })
            }
        }
        if(!this._running){
            return
        }
        this._masks = masks
        this._scanSize = { width, height }

        // The translations take the colour of the letters of the original (and follow it when it changes: the chosen line of a menu)
        for(const element of this._elements){
            if(element.state !== 'text' || element.segments.length === 0 || element.rect.w <= 0 || element.rect.h <= 0){
                continue
            }
            const mask = masks[this._polarities.indexOf(element.polarity)]
            if(mask === undefined){
                continue
            }
            for(const segment of element.segments){
                // The place of the segment in the picture that was looked at (the element knows both its places)
                const box: Region = {
                    x: element.scan.x + (segment.rect.x - element.rect.x) / element.rect.w * element.scan.w,
                    y: element.scan.y + (segment.rect.y - element.rect.y) / element.rect.h * element.scan.h,
                    w: segment.rect.w / element.rect.w * element.scan.w,
                    h: segment.rect.h / element.rect.h * element.scan.h,
                }
                const color = inkColor(pixels, mask.mask, width, height, box, element.polarity)
                // The background of the plate follows the picture behind the text (smoothed, now and then): the text is not read again for it
                const followedAt = Date.now()
                if(segment.baseAt === undefined || followedAt - segment.baseAt >= BACKGROUND_EVERY_MS){
                    const behind = backgroundColor(pixels, mask.mask, width, height, box)
                    const letters = segment.ink ?? (cssColor(segment.foreground) as [number, number, number] | null)
                    if(behind !== null && segment.behindRef === undefined){
                        // The first look is the reference: the colours of the reading are kept, only a change of the picture behind the text moves them
                        segment.behindRef = behind
                        segment.base0 = segment.base ?? segment.background
                    } else if(behind !== null && segment.behindRef !== undefined && segment.base0 !== undefined && letters !== null){
                        const shift = [behind[0] - segment.behindRef[0], behind[1] - segment.behindRef[1], behind[2] - segment.behindRef[2]]
                        if(Math.hypot(shift[0], shift[1], shift[2]) > BACKGROUND_CHANGED){
                            const clamp = (v: number) => Math.max(0, Math.min(255, Math.round(v)))
                            const target: [number, number, number] = [clamp(segment.base0[0] + shift[0]), clamp(segment.base0[1] + shift[1]), clamp(segment.base0[2] + shift[2])]
                            const was = segment.base ?? segment.background
                            const blended: [number, number, number] = [clamp(was[0] * 0.5 + target[0] * 0.5), clamp(was[1] * 0.5 + target[1] * 0.5), clamp(was[2] * 0.5 + target[2] * 0.5)]
                            segment.base = blended
                            const followed = plateColors(blended, letters)
                            segment.background = followed.background
                            segment.foreground = followed.foreground
                        } else if(segment.base !== undefined && Math.hypot(segment.base[0] - segment.base0[0], segment.base[1] - segment.base0[1], segment.base[2] - segment.base0[2]) > 6 && letters !== null){
                            // the picture is back as it was: the plate too
                            segment.base = segment.base0
                            const followed = plateColors(segment.base0, letters)
                            segment.background = followed.background
                            segment.foreground = followed.foreground
                        }
                    }
                    segment.baseAt = followedAt
                }
                // The colour of the letters must be on the same side of the background as the letters the reader saw (a text in light letters on a dark bar
                // is not turned into dark ones because the mask of the other polarity caught the bar itself)
                const luma = (c: number[]) => 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
                const under = segment.base ?? segment.background
                if(segment.fgLuma0 === undefined){
                    const seen = cssColor(segment.foreground)
                    segment.fgLuma0 = seen === null ? -1 : luma(seen)
                }
                if(color !== null && under !== undefined && segment.fgLuma0 >= 0 && (segment.fgLuma0 > luma(under)) !== (luma(color) > luma(under))){
                    continue
                }
                if(color === null || (segment.ink !== undefined && Math.hypot(color[0] - segment.ink[0], color[1] - segment.ink[1], color[2] - segment.ink[2]) <= INK_CHANGED)){
                    continue
                }
                segment.ink = color
                segment.base = segment.base ?? segment.background
                const colors = plateColors(segment.base, color)
                segment.background = colors.background
                segment.foreground = colors.foreground
            }
        }

        const now = Date.now()
        const manual = this._options.getManualRegions()
        const toScreen = (b: { x: number; y: number; w: number; h: number }): Region => videoRectToScreen(video.videoWidth, video.videoHeight, box.width, box.height, settings.fit, {
            x: visible.sx + b.x * visible.sw,
            y: visible.sy + b.y * visible.sh,
            w: b.w * visible.sw,
            h: b.h * visible.sh,
        })

        // The words of a big title are far apart ("H U N T E R  F O R U M S"): they are found as separate pieces and are put
        // together here, so that the title is translated as one phrase ("Hunter Forums" is "Форум мисливців", not "Мисливець Форуми")
        const pieces = this._joinTitles(blocks, toScreen)
        this._lastPieces = pieces.map(piece => ({ x: piece.x, y: piece.y, w: piece.w, h: piece.h, polarity: piece.polarity, role: piece.role, score: piece.score }))

        const taken = new Set<Element>()
        for(const block of pieces){
            const rect = toScreen(block)

            // The places the player has taken are not ours
            if(manual.some(region => region.enabled !== false && inside(rect, { x: region.x - 0.01, y: region.y - 0.01, w: region.w + 0.02, h: region.h + 0.02 }) > 0.02)){
                continue
            }

            const scan: Region = { x: block.x, y: block.y, w: block.w, h: block.h }
            const panel = block.panel === undefined ? undefined : toScreen(block.panel)

            // The same element as before: about the same place and size (a shift is followed: a text that floats, but never as far
            // as the next row of a list: that is another element)
            let best: Element | null = null
            let bestDistance = Infinity
            for(const element of this._elements){
                if(taken.has(element)){
                    continue
                }
                const same = (inside(rect, element.rect) > 0.6 && inside(element.rect, rect) > 0.4) || (similarSize(rect, element.rect) && distance(rect, element.rect) <= Math.max(0.015, Math.min(rect.h, element.rect.h) * 0.9))
                if(same){
                    const gap = distance(rect, element.rect)
                    if(gap < bestDistance){
                        best = element
                        bestDistance = gap
                    }
                }
            }

            // A text that scrolled: the place says it is another element (or none), the pixels of the letters say whose text it is
            const scrolled = this._scrolled(best, block.polarity, scan, rect, taken)
            let followRect = rect
            let clipped = false
            if(scrolled !== null){
                if(scrolled.edge === null){
                    this._relocate(scrolled.element, rect, scan)
                } else {
                    // The edge of the window cuts the line: the plate keeps the size of the whole line and is cut like it
                    const whole = scrolled.element.rect
                    followRect = { x: rect.x, w: whole.w, h: whole.h, y: scrolled.edge === 'top' ? rect.y : rect.y + rect.h - whole.h }
                    this._relocate(scrolled.element, followRect, scan)
                    scrolled.element.clip = rect
                    clipped = true
                    scrollStats.clipped++
                }
                best = scrolled.element
            }

            if(best === null){
                if(this._elements.length >= MAX_ELEMENTS){
                    continue
                }
                best = {
                    id: this._nextId++, rect, scan, polarity: block.polarity, role: block.role, plate: block.plate, panel, score: block.score,
                    seen: 0, missed: 0, state: 'new', dirty: false, dirtySince: now, reading: false, readAt: 0, deadUntil: 0, deadCount: 0,
                    sig: null, sigSum: 0, segments: [], detected: block.detected === true,
                }
                this._elements.push(best)
            }
            taken.add(best)
            this._follow(best, followRect, scan, panel, block.role, block.plate, block.polarity, block.score, now, clipped)
        }

        // Elements that were not found this time
        for(const element of this._elements){
            if(!taken.has(element)){
                element.missed++
                element.seen = 0

                // The text is not there any more (the scene has changed, a menu was closed): the translation is taken away at
                // once, not after a few more looks
                if(element.state === 'text' && element.sig !== null && element.sigSum > 0){
                    const mask = this._masks[this._polarities.indexOf(element.polarity)]
                    if(mask !== undefined){
                        const current = cellsOf(mask.mask, width, height, element.scan)
                        const difference = coreDifference(element.sig, current.cells)
                        // Not found where it was and the letters there are not its letters: it is gone (a look that comes seldom must not leave a stale plate for several looks)
                        if(current.sum < element.sigSum * 0.5 || difference > 0.55 || (element.missed >= 2 && difference > 0.4)){
                            this._park(element)
                            element.segments = []
                            element.state = 'idle'
                        }
                    }
                }
            }
        }
        this._elements = this._elements.filter(element => element.missed < (element.state === 'dead' ? DEAD_DROP_MISSES : DROP_MISSES) || element.reading)
        this._parked = this._parked.filter(entry => now - entry.at < PARKED_MAX_AGE_MS)

        this._emit()
        this._report()
    }

    /** An element that has gone out of sight is remembered with its translation (a text that scrolls away and back is not read again). */
    _park(element: Element) {
        if(element.state !== 'text' || element.sig === null || element.sigGrid === undefined || element.segments.length === 0){
            return
        }
        this._parked = this._parked.filter(entry => entry.element.id !== element.id)
        scrollStats.parked++
        this._parked.push({ element: { ...element, segments: element.segments.map(segment => ({ ...segment })), reading: false, dirty: false, missed: 0 }, at: Date.now() })
        if(this._parked.length > PARKED_MAX){
            this._parked.shift()
        }
    }

    /**
     * Is the text at this place (found by the scout) the text of an element that was somewhere else a moment ago (it scrolled), or one that went out
     * of sight earlier (it scrolled back)? Gives that element (a remembered one is put back among the elements), or null when the place is not it.
     */
    _scrolled(placeBest: Element | null, polarity: Polarity, scan: Region, rect: Region, taken: Set<Element>): { element: Element; edge: 'top' | 'bottom' | null } | null {
        if((globalThis as any).DLT_NOSCROLL === true){
            return null
        }
        const mask = this._masks[this._polarities.indexOf(polarity)]
        if(mask === undefined){
            return null
        }
        const { width, height } = this._scanSize
        const coarse = (element: Element): number => {
            if(element.sig === null || element.sigGrid === undefined || element.sigSum <= 0){
                return 1
            }
            const current = cellsOf(mask.mask, width, height, scan, element.sigGrid)
            if(current.sum < element.sigSum * 0.7 || current.sum > element.sigSum * 1.4){
                return 1
            }
            return coreDifference(element.sig, current.cells)
        }
        // The piece is a part of the line of the element: the edge of a window cuts the line (the letters of the piece are much lower than the letters of the line)
        const extent = inkExtent(mask.mask, width, height, scan)
        const cutOf = (element: Element) => {
            if(extent === null || element.sigPatch === undefined){
                return false
            }
            const whole = element.sigPatch.bottom - element.sigPatch.top + 1
            const seen = extent.bottom - extent.top + 1
            const across = element.sigPatch.right - element.sigPatch.left + 1
            return seen < whole * CLIP_MAX && seen > whole * CLIP_MIN && extent.right - extent.left + 1 > across * 0.5 && extent.right - extent.left + 1 < across * 1.3
        }

        // The element that is at this place already and is the same text: nothing moved
        if(placeBest !== null && placeBest.state === 'text' && placeBest.sig !== null && !cutOf(placeBest) && coarse(placeBest) <= PLACE_MISMATCH){
            return null
        }

        let found = null as Element | null
        let foundEdge: 'top' | 'bottom' | null = null
        let foundScore = SCROLL_PATCH
        let parked = -1
        const consider = (element: Element, index: number) => {
            const cut = cutOf(element)
            if(taken.has(element) || (element === placeBest && !cut) || element.reading || element.state !== 'text' || element.segments.length === 0 || element.sigPatch === undefined
                || element.polarity !== polarity || Math.abs(rect.x - element.rect.x) > SCROLL_MAX_DX || Math.abs(rect.y - element.rect.y) > SCROLL_MAX_DY + element.rect.h){
                return
            }
            if(cut && extent !== null){
                const top = patchMatchPartial(element.sigPatch, mask.mask, width, height, extent, 'top')
                const bottom = patchMatchPartial(element.sigPatch, mask.mask, width, height, extent, 'bottom')
                const score = Math.max(top, bottom)
                if(score >= CLIP_PATCH && score > foundScore){
                    found = element
                    foundScore = score
                    foundEdge = top >= bottom ? 'top' : 'bottom'
                    parked = index
                }
                return
            }
            if(!similarSize(rect, element.rect) || coarse(element) > SCROLL_COARSE){
                return
            }
            const score = patchMatch(element.sigPatch, mask.mask, width, height, scan)
            if(score > foundScore){
                found = element
                foundScore = score
                foundEdge = null
                parked = index
            }
        }
        for(const element of this._elements){
            consider(element, -1)
        }
        for(let i = 0; i < this._parked.length; i++){
            if(!this._elements.some(other => other.id === this._parked[i].element.id)){
                consider(this._parked[i].element, i)
            }
        }
        if(found === null){
            return null
        }
        scrollStats.moved++
        if(scrollStats.scores.length < 400){
            scrollStats.scores.push(Math.round(foundScore * 100) / 100)
        }
        const element: Element = found
        if(parked >= 0){
            scrollStats.restored++
            // The text is back: the element that was remembered takes the place (what stands there now, not read yet, gives way to it)
            this._parked.splice(parked, 1)
            element.seen = Math.max(element.seen, 2)
            element.missed = 0
            if(placeBest !== null && placeBest.state !== 'text' && !placeBest.reading){
                this._elements = this._elements.filter(other => other !== placeBest)
            }
            this._elements.push(element)
        }
        return { element, edge: foundEdge }
    }

    /** A text that scrolled: the element and its translation are moved to the new place. */
    _relocate(element: Element, rect: Region, scan: Region) {
        const dx = rect.x - element.rect.x
        const dy = rect.y - element.rect.y
        element.segments = element.segments.map(segment => ({ ...segment, rect: { ...segment.rect, x: segment.rect.x + dx, y: segment.rect.y + dy } }))
        this._noteMotion(element, dx, dy, Date.now())
        element.rect = rect
        element.scan = scan
        element.state = 'text'
        element.clip = undefined
    }

    /** How fast a text is moving (the way it went since the place was last confirmed, over the time that has passed), and whether it keeps going the same way. */
    _noteMotion(element: Element, dx: number, dy: number, now: number) {
        const seconds = Math.max(0.25, (now - (element.placedAt ?? now - 500)) / 1000)
        let vx = Math.max(-MOTION_MAX, Math.min(MOTION_MAX, dx / seconds))
        let vy = Math.max(-MOTION_MAX, Math.min(MOTION_MAX, dy / seconds))
        const before = element.motion
        const sustained = before !== undefined && now - before.at < 1200 && vx * before.vx + vy * before.vy > 0
        if(sustained && before !== undefined){
            vx = (vx + before.vx) / 2
            vy = (vy + before.vy) / 2
        }
        element.motion = { vx, vy, at: now, sustained }
    }

    /**
     * A picture of the game without the plates (taken another way, so its pixels are not the same as those of the live picture: it must
     * not replace them, or everything looks changed). It only answers one question: is the text under a plate still the same? An element
     * whose text pixels differ a lot from what was read is read again.
     */
    async verifyAgainst(pixels: Uint8ClampedArray, width: number, height: number): Promise<{ checked: number; changed: number; differences: number[]; rects: Region[] }> {
        // `rects`: where the text changed (fractions of the picture): the caller puts the new pixels of these places into the picture the
        // engine reads, which still holds the old ones under a plate
        const result = { checked: 0, changed: 0, differences: [] as number[], rects: [] as Region[] }
        if(!this._running){
            return result
        }
        const masks = await inkMasksInSlices(pixels, width, height, this._polarities, () => new Promise<void>(resolve => setTimeout(resolve, 0)))
        if(!this._running){
            return result
        }
        const now = Date.now()
        for(const element of this._elements){
            if(element.state !== 'text' || element.sig === null || element.sigGrid === undefined || element.reading || element.dirty){
                continue
            }
            const mask = masks[this._polarities.indexOf(element.polarity)]
            if(mask === undefined){
                continue
            }
            const current = cellsOf(mask.mask, width, height, element.scan, element.sigGrid)
            const difference = coreDifference(element.sig, current.cells)
            result.checked++
            result.differences.push(Math.round(difference * 100) / 100)
            if(difference > VERIFY_CHANGED || current.sum < element.sigSum * 0.35){
                element.dirty = true
                element.dirtySince = now
                result.changed++
                result.rects.push({ x: element.rect.x, y: element.rect.y, w: element.rect.w, h: element.rect.h })
            }
        }
        return result
    }

    /** Is there nothing between two pieces of text (no frame of a button, no icon)? Only then they can be one title. */
    _gapIsEmpty(a: { x: number; y: number; w: number; h: number; polarity: Polarity; score: number }, b: { x: number; y: number; w: number; h: number; polarity: Polarity; score: number }): boolean {
        const left = a.x <= b.x ? a : b
        const right = a.x <= b.x ? b : a
        const { width, height } = this._scanSize
        const x0 = Math.floor((left.x + left.w) * width)
        const x1 = Math.ceil(right.x * width)
        const y0 = Math.floor(Math.max(a.y, b.y) * height)
        const y1 = Math.ceil(Math.min(a.y + a.h, b.y + b.h) * height)
        if(x1 <= x0 || y1 <= y0){
            return true
        }

        // The text pixels of the kind of the better piece (the other kind is the plain background next to the letters)
        const mask = this._masks[this._polarities.indexOf(a.score >= b.score ? a.polarity : b.polarity)]
        if(mask === undefined){
            return true
        }
        // A line that runs through the gap (the edge of a button, a divider) tells the pieces are not one phrase; so does a lot of ink
        // (an icon). A little ink at the edges of the letters is only the glow around them.
        let count = 0
        let line = false
        for(let x = x0; x < x1; x++){
            let column = 0
            for(let y = y0; y < y1; y++){
                if(mask.mask[y * width + x] === 1){
                    column++
                }
            }
            count += column
            if(column >= GAP_LINE * (y1 - y0)){
                line = true
            }
        }
        return !line && count / ((x1 - x0) * (y1 - y0)) < GAP_INK
    }

    /** Big pieces of text that stand on one row, close to each other and about as big, are one piece (a title in several words). */
    _joinTitles<T extends { x: number; y: number; w: number; h: number; score: number; polarity: Polarity }>(blocks: T[], toScreen: (b: { x: number; y: number; w: number; h: number }) => Region): T[] {
        const list = blocks.slice()
        let joined = true

        while(joined){
            joined = false
            for(let a = 0; a < list.length && !joined; a++){
                for(let b = a + 1; b < list.length && !joined; b++){
                    const first = toScreen(list[a])
                    const second = toScreen(list[b])
                    const high = Math.max(first.h, second.h)
                    const low = Math.min(first.h, second.h)
                    const overlap = Math.min(first.y + first.h, second.y + second.h) - Math.max(first.y, second.y)
                    const gap = Math.max(first.x, second.x) - Math.min(first.x + first.w, second.x + second.w)
                    if(low >= TITLE_MIN_HEIGHT && high <= TITLE_MAX_HEIGHT && low / high >= 0.6 && overlap >= 0.7 * low && gap <= TITLE_GAP * high){
                        const x = Math.min(list[a].x, list[b].x)
                        const y = Math.min(list[a].y, list[b].y)
                        const best = list[a].score >= list[b].score ? list[a] : list[b]
                        list[a] = { ...best, x, y, w: Math.max(list[a].x + list[a].w, list[b].x + list[b].w) - x, h: Math.max(list[a].y + list[a].h, list[b].y + list[b].h) - y, score: Math.max(list[a].score, list[b].score) }
                        list.splice(b, 1)
                        joined = true
                    }
                }
            }
        }
        return list
    }

    /** An element was found again: its place is followed, and its text pixels say whether it has to be read (again). */
    _follow(element: Element, rect: Region, scan: Region, panel: Region | undefined, role: BlockRole, plate: boolean, polarity: Polarity, score: number, now: number, clipped = false) {
        element.missed = 0
        element.seen++
        element.score = Math.max(element.score * 0.9, score)
        element.role = role
        element.plate = plate
        element.panel = panel
        element.polarity = polarity

        // A text that moves (it floats in the world): its translation goes with it
        const dx = rect.x - element.rect.x
        const dy = rect.y - element.rect.y
        if((Math.abs(dx) >= MOVE_TOLERANCE || Math.abs(dy) >= MOVE_TOLERANCE) && similarSize(rect, element.rect)){
            element.segments = element.segments.map(segment => ({ ...segment, rect: { ...segment.rect, x: segment.rect.x + dx, y: segment.rect.y + dy } }))
            this._noteMotion(element, dx, dy, now)
            element.rect = rect
        } else if(!similarSize(rect, element.rect)){
            element.rect = rect
        }
        element.scan = scan
        element.placedAt = now

        // A line that the edge of its window cuts is not read, nor compared with what was read (its pixels are a part of the line): it is only followed
        if(clipped){
            element.lastCells = undefined
            element.changing = false
            return
        }
        element.clip = undefined

        const polarityIndex = this._polarities.indexOf(element.polarity)
        const mask = this._masks[polarityIndex]
        if(mask === undefined){
            return
        }
        const current = cellsOf(mask.mask, this._scanSize.width, this._scanSize.height, scan)

        // Is the text still changing (typed letter by letter)?
        const moved = element.lastCells !== undefined && coreDifference(element.lastCells, current.cells) > STILL_DIFF
        element.lastCells = current.cells
        if(moved){
            if(element.changing !== true){
                element.changingSince = now
            }
            this._lookChanged = true
        }
        element.changing = moved

        if(element.state === 'new'){
            if(!element.dirty && element.seen >= this._need(element)){
                element.dirty = true
                element.dirtySince = now
            }
            return
        }

        if(element.sig === null){
            return
        }

        const difference = coreDifference(element.sig, current.cells)

        if(element.state === 'dead'){
            if((now >= element.deadUntil || difference > REVIVE_CHANGE) && !element.dirty){
                element.dirty = true
                element.dirtySince = now
            }
            return
        }

        // The text is gone (what was read is not there any more): the translation is taken away at once
        if(element.sigSum > 0 && current.sum < element.sigSum * GONE){
            if(element.segments.length > 0){
                element.segments = []
                element.state = 'idle'
            }
            return
        }

        // Most of the text pixels are others than what was read: the place is read at once, and the plate stays until the reading says what is there now (a button
        // that is highlighted changes its look, not its text: its plate must not blink). A reading that finds nothing takes the plate away.
        if(difference > HIDE_CHANGED && element.segments.length > 0){
            element.strongChange = true
            element.readAt = 0
        }

        // The text is back after it had gone (or the text pixels changed): the element is read again
        if((difference > CHANGED || element.state === 'idle') && !element.dirty){
            element.dirty = true
            element.dirtySince = now
        }
    }

    /** On how many looks in a row an element has to be found before it is read (a subtitle at once, the rest of the interface a little later). */
    _need(element: Element): number {
        let need = element.role === 'subtitle' ? 1 : (element.plate || element.role === 'notice' ? 2 : 3)
        if(centralness(element.rect) > 0){
            need = Math.min(need, 2)
        }
        // A piece much smaller than the lines next to it is the end of a line that an edge cuts (a text that scrolls): it is read when it has stayed
        const heights = this._elements.filter(other => other !== element && other.state === 'text' && Math.abs(other.rect.x - element.rect.x) < 0.04).map(other => other.rect.h).sort((a, b) => a - b)
        if(heights.length >= 2 && element.rect.h < heights[Math.floor(heights.length / 2)] * SLIVER_SHARE){
            need += SLIVER_LOOKS
        }
        // A picture with very many pieces of text (a busy moving background) is looked at more carefully before a piece is read: it is mostly false alarms
        return need + (element.polarity === 'dark' ? 1 : 0) + Math.floor(this._elements.length / 14)
    }

    // ----------------------------------------------------------------------------------------------- reading

    async _readLoop() {
        if(!this._running){
            return
        }

        let next: Element | null = null
        if(this._ready){
            next = this._pick()
        }

        const started = Date.now()
        if(next !== null){
            try {
                await this._readElement(next)
            } catch(error) {
                console.warn('Translator: reading failed:', error)
                next.reading = false
            }
        }

        if(this._running){
            // The reader may take only a part of the time (the battery, the game): the longer a reading took, the longer the rest
            const duty = this._options.readDuty?.() ?? 1
            const rest = duty > 0 && duty < 1 ? Math.round((Date.now() - started) * (1 / duty - 1)) : 0
            this._readTimer = setTimeout(() => this._readLoop(), next === null ? 120 : rest)
        }
    }

    /** The element that is read next: new text of subtitles and of the middle of the screen first, then the rest. */
    _pick(): Element | null {
        const now = Date.now()
        let best: Element | null = null
        let bestScore = -Infinity

        for(const element of this._elements){
            if(!element.dirty || element.reading || element.missed > 0){
                continue
            }
            const gap = (element.state === 'text' ? SHOWN_READ_GAP_MS : READ_GAP_MS) * (this._options.speed?.() ?? 1)
            if(now - element.readAt < gap){
                continue
            }
            if(element.changing === true && element.rect.w * element.rect.h < TYPING_MAX_AREA && now - (element.changingSince ?? now) < SETTLE_LIMIT_MS){
                continue
            }

            const score = (element.role === 'subtitle' ? 4 : 0) + centralness(element.rect) * 3 + (element.plate ? 1 : 0) + (element.state === 'new' ? 1.5 : 0) + Math.min(3, (now - element.dirtySince) / 1500)
            if(score > bestScore){
                best = element
                bestScore = score
            }
        }
        return best
    }

    async _readElement(element: Element) {
        const video = this._options.getVideo()
        if(video === null || video.readyState < 2){
            return
        }

        const box = this._options.getBox()
        const settings = this._options.getSettings()

        // The text pixels at this moment: this is what is read
        const polarityIndex = this._polarities.indexOf(element.polarity)
        const mask = this._masks[polarityIndex]
        const current = mask === undefined ? null : cellsOf(mask.mask, this._scanSize.width, this._scanSize.height, element.scan)

        // The place of the element a little bigger, so that the edges of the letters are not cut
        const padY = Math.max(0.006, element.rect.h * 0.3)
        const padX = Math.max(0.006, element.rect.h * 0.4)
        const region: Region = { x: Math.max(0, element.rect.x - padX), y: Math.max(0, element.rect.y - padY), w: element.rect.w + 2 * padX, h: element.rect.h + 2 * padY }
        const source = regionToSource(video.videoWidth, video.videoHeight, box.width, box.height, settings.fit, region)
        if(source === null){
            element.dirty = false
            return
        }

        element.dirty = false
        element.reading = true
        element.readAt = Date.now()
        const strong = element.strongChange === true
        element.strongChange = false

        const crop = document.createElement('canvas')
        crop.width = Math.max(8, Math.round(source.sw))
        crop.height = Math.max(8, Math.round(source.sh))
        const context = crop.getContext('2d')
        if(context === null){
            element.reading = false
            return
        }
        context.drawImage(video, source.sx, source.sy, source.sw, source.sh, 0, 0, crop.width, crop.height)

        const job: any = {
            canvas: crop,
            source,
            config: { region, mode: element.role === 'subtitle' ? 'dialog' : 'task', color: 'auto', source: settings.source, target: settings.target, fit: settings.fit },
        }
        // The text colour the scout found first; when nothing is read, the other one (white text on a light bar looks dark)
        // and, when that does not work, the picture in grey (thin or faint letters are lost when it is cut into black and white)
        const other: Polarity = element.polarity === 'light' ? 'dark' : 'light'
        const attempts: Array<[Polarity, boolean | 'stretched' | 'color']> = [[element.polarity, false], [other, false]]
        // The modern reader looks at the natural picture first
        if(getReaderMode() !== 'tesseract' && ppocrReady()){
            // The modern reader looks once at the natural picture; for a place the scout found the black-and-white way of Tesseract is tried after it
            // (the other ways, with the other polarity and in grey, cost much time and are left to the classic reader)
            attempts.length = 0
            attempts.push([element.polarity, 'color'])
            if(element.detected !== true){
                attempts.push([element.polarity, false])
            }
        }
        if(element.score >= RAW_MIN_SCORE){
            attempts.push([element.polarity, true], [element.polarity, 'stretched'])
        }
        let result: Awaited<ReturnType<TranslatorEngine['_read']>> = null
        let resultValue = 0
        for(const [polarity, raw] of attempts){
            if(!this._running){
                break
            }
            const attempt = await this._toolbox._read(job, polarity, raw)
            if(attempt !== null && attempt.segments.length > 0){
                // How much of the place the text that was read covers: a reading of one line out of a paragraph is a poor one
                const coverage = this._coverage(element, attempt, video, box, settings.fit)
                const value = attempt.confidence * (0.4 + 0.6 * coverage)
                if(result === null || value > resultValue){
                    result = attempt
                    resultValue = value
                }
                if(attempt.confidence >= GOOD_CONFIDENCE && coverage >= GOOD_COVERAGE){
                    break
                }
            }
        }
        crop.width = 0
        crop.height = 0
        element.reading = false

        if(!this._running || !this._elements.includes(element)){
            return
        }

        // The text changed while it was being read (it was typed on): what was read is a part of it, and it is not shown. The next
        // reading is called at once (the gap between two readings does not apply).
        if(current !== null && element.lastCells !== undefined && Date.now() - (element.changingSince ?? Date.now()) < SETTLE_LIMIT_MS
            && (element.lastCells.length !== current.cells.length || coreDifference(current.cells, element.lastCells) > STILL_DIFF * 1.5)){
            element.dirty = true
            element.dirtySince = Date.now()
            element.readAt = 0
            return
        }

        if(current !== null){
            element.sig = current.cells
            element.sigSum = current.sum
            element.sigGrid = { gx: current.gx, gy: current.gy }
            element.sigPatch = mask === undefined ? undefined : (patchOf(mask.mask, this._scanSize.width, this._scanSize.height, element.scan) ?? undefined)
        }

        // A place that only the detector found (a picture, a rock, a pattern that looks like letters) must give a real text: a short word is believed
        // only when it is a word of the interface of a game ("Play", "Back")
        if(result !== null && element.detected === true){
            const letters = result.segments.reduce((sum, segment) => sum + (segment.text.match(LETTER) || []).length, 0)
            if(letters < 5 && glossaryTranslation(result.segments.map(segment => segment.text).join(' '), this._options.getSettings().source, 'uk') === null){
                result = null
            }
        }

        // Short words only ("and i as") from a place that only the detector found: the pattern of an ornament, not a sentence
        if(result !== null && element.detected === true){
            const words = result.segments.flatMap(segment => segment.text.split(/\s+/)).filter(word => /\p{L}/u.test(word))
            const letters = result.segments.reduce((sum, segment) => sum + (segment.text.match(LETTER) || []).length, 0)
            if(words.length >= 3 && words.every(word => word.replace(/[^\p{L}]/gu, '').length <= 3) && letters < 10){
                result = null
            }
        }

        // A few letters that the reader is not sure about are the pattern of an icon (a button of the pad, a symbol), not a word
        if(result !== null && result.confidence < MIN_CONFIDENCE){
            result = null
        }
        if(result !== null && result.segments.length > 0 && result.confidence < SHORT_TEXT_CONFIDENCE){
            const letters = result.segments.reduce((sum, segment) => sum + (segment.text.match(LETTER) || []).length, 0)
            if(letters <= SHORT_TEXT_LETTERS){
                result = null
            }
        }

        if((result === null || result.segments.length === 0) && strong && element.segments.length > 0){
            // The text changed much and nothing can be read there now: the old translation does not belong here
            element.segments = []
            element.state = 'idle'
            this._emit()
            return
        }

        if(result === null || result.segments.length === 0){
            // Nothing readable: a false alarm (a picture, a decoration) is left alone for a while, longer every time
            if(element.state === 'text' && element.segments.length > 0 && element.deadCount < KEEP_FAILURES){
                // It had text before and a reading may fail now and then: the plate stays and the reading is tried again
                element.deadCount++
                element.dirty = true
                element.dirtySince = Date.now()
                return
            }
            if(element.state === 'text' && element.deadCount === 0){
                element.deadCount = 1
                element.dirty = true
                element.dirtySince = Date.now()
                return
            }
            element.state = 'dead'
            element.segments = []
            element.deadCount++
            element.deadUntil = Date.now() + DEAD_MS * Math.pow(2, Math.min(4, element.deadCount - 1))
            this._emit()
            return
        }
        element.deadCount = 0

        // The lines that were read: the ones that are the same text as before keep their translation (nothing blinks)
        const target = settings.target
        const old = element.segments
        const used = new Set<Segment>()
        const segments: Segment[] = []
        const pendingTranslations: Array<Promise<void>> = []

        for(const raw of result.segments){
            const screen = videoRectToScreen(video.videoWidth, video.videoHeight, box.width, box.height, settings.fit, raw.rect)
            const lineHeight = videoRectToScreen(video.videoWidth, video.videoHeight, box.width, box.height, settings.fit, { x: 0, y: 0, w: 0, h: raw.lineHeight }).h
            const template = templateKey(raw.text)
            const same = old.find(segment => !used.has(segment) && (segment.template === template || isSameReading(template, segment.template)))

            if(same !== undefined){
                used.add(same)
                const moved = Math.abs(same.rect.x - screen.x) >= MOVE_TOLERANCE || Math.abs(same.rect.y - screen.y) >= MOVE_TOLERANCE || Math.abs(same.rect.w - screen.w) >= MOVE_TOLERANCE * 2
                segments.push(moved ? { ...same, rect: screen, lineHeight } : same)
                continue
            }

            const segment: Segment = {
                original: raw.text,
                translated: '',
                rect: screen,
                lineHeight,
                background: raw.background,
                foreground: raw.foreground,
                centered: raw.centered,
                template,
                shownAt: Date.now(),
            }
            segments.push(segment)
            pendingTranslations.push(this._translate(raw.text, target, settings.source).then((translated) => { segment.translated = translated }))
        }

        try {
            await Promise.all(pendingTranslations)
        } catch(error) {
            console.warn('Translator: translation failed:', error)
            this._options.onStatus?.('error')
            // Again in a few seconds
            element.dirty = true
            element.dirtySince = Date.now()
            element.readAt = Date.now() + 2500
            return
        }

        if(!this._running || !this._elements.includes(element)){
            return
        }

        this._options.onStatus?.('ready')
        element.segments = segments.filter(segment => segment.translated !== '')
        element.state = element.segments.length > 0 ? 'text' : 'dead'
        this._emit()
    }

    // ----------------------------------------------------------------------------------------------- translating

    _translate(text: string, target: string, source: string): Promise<string> {
        const key = target + '|' + normalizeText(text)
        const known = this._cache.get(key)
        if(known !== undefined){
            return Promise.resolve(known)
        }

        let pending = this._pending.get(key)
        if(pending === undefined){
            pending = this._options.translate(toTranslatable(text), target, source).then((answer) => {
                this._pending.delete(key)
                let translated = typeof answer === 'string' ? answer : answer.text
                if(translated === ''){
                    translated = text
                }
                this._remember(key, translated)

                // A draft that gets better: the better text replaces it where it is shown (while it is young), and in the memory
                if(typeof answer !== 'string' && answer.better !== undefined){
                    const draft = translated
                    answer.better.then((better) => {
                        if(better === null || better === '' || better === draft || !this._running){
                            return
                        }
                        this._remember(key, better)
                        const now = Date.now()
                        let changed = false
                        this._elements.forEach((element) => {
                            element.segments.forEach((segment) => {
                                if(segment.translated === draft && target + '|' + normalizeText(segment.original) === key && now - segment.shownAt < BETTER_MS){
                                    segment.translated = better
                                    changed = true
                                }
                            })
                        })
                        if(changed){
                            this._emit()
                        }
                    }).catch(() => { /* the draft stays */ })
                }
                return translated
            }, (error) => {
                this._pending.delete(key)
                throw error
            })
            this._pending.set(key, pending)
        }
        return pending
    }

    _remember(key: string, translated: string) {
        if(this._cache.size > 400){
            this._cache.delete(this._cache.keys().next().value as string)
        }
        this._cache.set(key, translated)
    }

    // ----------------------------------------------------------------------------------------------- showing

    /** What is on screen: the translations of the elements that have one. It is passed on only when it has changed. */
    _emit() {
        const settings = this._options.getSettings()
        const results: ElementResult[] = []

        for(const element of this._elements){
            if(element.state !== 'text' || element.segments.length === 0){
                continue
            }
            const moving = element.motion !== undefined && Date.now() - element.motion.at < 1000 ? element.motion : undefined
            const segments: TranslatedSegment[] = element.segments.map(segment => ({
                original: segment.original, translated: segment.translated, rect: segment.rect, lineHeight: segment.lineHeight,
                background: segment.background, foreground: segment.foreground, centered: segment.centered, shownAt: segment.shownAt,
                ...(moving !== undefined ? { motion: moving } : {}),
                ...(element.clip !== undefined ? { clip: element.clip } : {}),
            }))
            results.push({
                id: element.id,
                role: element.role,
                result: {
                    original: segments.map(segment => segment.original).join(' '),
                    translated: segments.map(segment => segment.translated).join('\n'),
                    segments,
                    ...(settings.panels !== false && element.panel !== undefined ? { bounds: element.panel } : {}),
                },
            })
        }

        // The same text can be found twice: as a line of its own and inside a bigger piece (a speech box with its name tab and its text,
        // when the frame of the box is drawn by hand and moves a little). One plate per place: the one of the tighter, smaller piece stays.
        const areas = new Map<number, number>(this._elements.map(element => [element.id, element.rect.w * element.rect.h]))
        const flat = results.flatMap(item => item.result.segments.map(segment => ({ item, segment })))
        const dropped = new Set<TranslatedSegment>()
        for(const a of flat){
            for(const b of flat){
                if(a === b || a.item === b.item || dropped.has(b.segment)){
                    continue
                }
                const smaller = Math.min(a.segment.rect.w * a.segment.rect.h, b.segment.rect.w * b.segment.rect.h)
                if(smaller <= 0 || inside(a.segment.rect, b.segment.rect) * (a.segment.rect.w * a.segment.rect.h) / smaller < 0.5){
                    continue
                }
                // only the same text : another text at the same place is not a copy
                const textA = a.segment.original.toUpperCase().replace(/[^\p{L}\p{N}]/gu, '')
                const textB = b.segment.original.toUpperCase().replace(/[^\p{L}\p{N}]/gu, '')
                if(textA.length === 0 || textB.length === 0 || textA !== textB){
                    continue
                }
                const areaA = areas.get(a.item.id) ?? 0
                const areaB = areas.get(b.item.id) ?? 0
                if(areaA > areaB || (areaA === areaB && a.item.id > b.item.id)){
                    dropped.add(a.segment)
                    break
                }
            }
        }
        if(dropped.size > 0){
            for(const item of results){
                item.result.segments = item.result.segments.filter(segment => !dropped.has(segment))
                item.result.original = item.result.segments.map(segment => segment.original).join(' ')
                item.result.translated = item.result.segments.map(segment => segment.translated).join('\n')
            }
            results.splice(0, results.length, ...results.filter(item => item.result.segments.length > 0))
        }

        const key = JSON.stringify(results.map(item => [item.id, item.result.translated, item.result.segments.map(segment => [Math.round(segment.rect.x * 1000), Math.round(segment.rect.y * 1000), Math.round(segment.rect.w * 1000), segment.background[0] >> 3, segment.background[1] >> 3, segment.background[2] >> 3, segment.clip === undefined ? 0 : Math.round(segment.clip.y * 1000) * 1000 + Math.round(segment.clip.h * 1000)])]))
        if(key !== this._lastKey){
            this._lastKey = key
            this._options.onResults(results)
        }
    }

    _report() {
        if(this._options.onFrames === undefined){
            return
        }

        const frames: AutoFrame[] = this._elements.filter(element => element.missed === 0).map((element, index) => ({
            rect: element.rect,
            role: element.role,
            state: element.state === 'text' ? 'text' : (element.state === 'dead' ? 'rejected' : (element.reading || element.dirty ? 'reading' : (element.state === 'idle' ? 'idle' : 'candidate'))),
            slot: index,
            plate: element.plate,
        }))
        this._options.onFrames(frames)
    }

    /** How much of the place of an element the text of a reading covers (0..1). */
    _coverage(element: Element, reading: { segments: Array<{ rect: { x: number; y: number; w: number; h: number } }> }, video: HTMLVideoElement, box: { width: number; height: number }, fit: FitMode): number {
        let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity
        for(const segment of reading.segments){
            const screen = videoRectToScreen(video.videoWidth, video.videoHeight, box.width, box.height, fit, segment.rect)
            x0 = Math.min(x0, screen.x)
            y0 = Math.min(y0, screen.y)
            x1 = Math.max(x1, screen.x + screen.w)
            y1 = Math.max(y1, screen.y + screen.h)
        }
        const area = element.rect.w * element.rect.h
        if(area <= 0 || x1 <= x0 || y1 <= y0){
            return 1
        }
        return Math.min(1, ((x1 - x0) * (y1 - y0)) / area)
    }

    /** What every element is doing (for tests). */
    getDebug(): string[] {
        return this._elements.map(element => '#' + element.id + ' ' + element.state + ' ' + element.role + ' seen' + element.seen + ' miss' + element.missed + ' x' + element.rect.x.toFixed(2) + ' y' + element.rect.y.toFixed(2) + ' w' + element.rect.w.toFixed(2) + ' ' + (element.segments.map(segment => segment.original).join(' / ') || '-'))
    }

    /** How many elements there are (for the debug view). */
    getCount(): number {
        return this._elements.length
    }
}
