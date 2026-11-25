// The text reader library (tesseract.js) starts a job on its worker with a call that fails when the worker was shut down a moment before: "Cannot read properties of null (reading
// 'postMessage')". That happens when a stream is closed (or the language is changed) while a reading goes on. The failed promise is inside the library, where nothing of ours can
// catch it, and it reached the page as an unhandled rejection that the page showed in an alert. Nobody waits for that reading any more, so the error is dropped: but only for a while
// after the reader was shut down by us, so that the same error from anything else is still shown.
let lastShutdownAt = 0

/** The reader is being shut down now. */
export function markReaderShutdown(): void {
    lastShutdownAt = Date.now()
}

export function isReaderShutdownError(reason: unknown): boolean {
    return /Cannot read propert(y|ies) of null \(reading 'postMessage'\)/.test(String((reason as any)?.message ?? reason))
}

/** Is this unhandled rejection the harmless one of a reader that was shut down a moment ago? */
export function isHarmlessReaderRejection(reason: unknown): boolean {
    return isReaderShutdownError(reason) && Date.now() - lastShutdownAt < 60000
}
