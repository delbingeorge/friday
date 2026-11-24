// friday-audio: mic + speaker through macOS Voice Processing I/O (Apple's echo
// cancellation), so Friday can hear you over its own voice on laptop speakers.
//
// Protocol (all integers little-endian), each message = type(1) len(u32) payload:
//   stdout  'R' ready            | 'M' 512 x int16 mic samples @16 kHz mono
//           'D' u32 id: buffer finished playing
//   stdin   'P' u32 id + float32 samples @24 kHz mono: queue for playback
//           'S' stop playback and drop everything queued | 'Q' quit
//           'B' u8: 1 = bypass voice processing (idle), 0 = enable it (Friday is talking)
import AVFoundation
import Foundation

let outLock = NSLock()
func send(_ type: UInt8, _ payload: Data = Data()) {
    var msg = Data([type])
    var n = UInt32(payload.count).littleEndian
    withUnsafeBytes(of: &n) { msg.append(contentsOf: $0) }
    msg.append(payload)
    outLock.lock()
    defer { outLock.unlock() }
    do { try FileHandle.standardOutput.write(contentsOf: msg) } catch { exit(0) }  // parent gone
}

func fail(_ s: String) -> Never {
    FileHandle.standardError.write(("friday-audio: " + s + "\n").data(using: .utf8)!)
    exit(2)
}

let engine = AVAudioEngine()
let input = engine.inputNode
// Instantiate the output side first so voice processing is applied to both ends.
_ = engine.outputNode
_ = engine.mainMixerNode
do {
    try input.setVoiceProcessingEnabled(true)
    if !engine.outputNode.isVoiceProcessingEnabled { try engine.outputNode.setVoiceProcessingEnabled(true) }
} catch { fail("voice processing unavailable: \(error)") }
if #available(macOS 14.0, *) {
    // Don't turn down other apps' audio (music) while Friday is running.
    input.voiceProcessingOtherAudioDuckingConfiguration =
        AVAudioVoiceProcessingOtherAudioDuckingConfiguration(enableAdvancedDucking: false, duckingLevel: .min)
}

let player = AVAudioPlayerNode()
engine.attach(player)
let playFmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 24000, channels: 1, interleaved: false)!
engine.connect(player, to: engine.mainMixerNode, format: playFmt)

// Voice processing delivers several identical Float32 channels (5 on a MacBook Air)
// at 48 kHz. AVAudioConverter turned that into silence, so resample channel 0 by
// hand: average each 1/16000 s window (a box low-pass, plenty for speech).
let inFmt = input.outputFormat(forBus: 0)
let step = inFmt.sampleRate / 16000.0
var phase = 0.0  // fractional input position carried across buffers
var acc: Float = 0
var accN = 0
var pending = [Int16]()
pending.reserveCapacity(4096)
input.installTap(onBus: 0, bufferSize: 1024, format: inFmt) { buf, _ in
    guard let ch = buf.floatChannelData else { return }
    let x = ch[0]
    for i in 0..<Int(buf.frameLength) {
        acc += x[i]
        accN += 1
        phase += 1
        if phase >= step {
            phase -= step
            let v = max(-1, min(1, acc / Float(accN)))
            pending.append(Int16(v * 32767))
            acc = 0
            accN = 0
        }
    }
    while pending.count >= 512 {
        let frame = Data(bytes: pending, count: 512 * 2)
        pending.removeFirst(512)
        send(UInt8(ascii: "M"), frame)
    }
}

NotificationCenter.default.addObserver(forName: .AVAudioEngineConfigurationChange, object: engine, queue: nil) { _ in
    // Headphones plugged in / device switched: restart on the new route.
    try? engine.start()
    player.play()
}

engine.prepare()
do { try engine.start() } catch { fail("engine start failed: \(error)") }
player.play()

func readExact(_ n: Int) -> Data? {
    var d = Data()
    while d.count < n {
        let chunk = FileHandle.standardInput.readData(ofLength: n - d.count)
        if chunk.isEmpty { return nil }
        d.append(chunk)
    }
    return d
}

Thread {
    while true {
        guard let head = readExact(5) else { exit(0) }
        let type = head[head.startIndex]
        let len = Int(head.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 1, as: UInt32.self) })
        guard let body = len > 0 ? readExact(len) : Data() else { exit(0) }
        switch type {
        case UInt8(ascii: "P"):
            let id = body.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 0, as: UInt32.self) }
            let count = (len - 4) / 4
            guard count > 0, let pcm = AVAudioPCMBuffer(pcmFormat: playFmt, frameCapacity: AVAudioFrameCount(count))
            else { send(UInt8(ascii: "D"), withUnsafeBytes(of: id) { Data($0) }); continue }
            pcm.frameLength = AVAudioFrameCount(count)
            _ = body.withUnsafeBytes { raw in
                memcpy(pcm.floatChannelData![0], raw.baseAddress! + 4, count * 4)
            }
            player.scheduleBuffer(pcm, completionCallbackType: .dataPlayedBack) { _ in
                send(UInt8(ascii: "D"), withUnsafeBytes(of: id) { Data($0) })
            }
            if !player.isPlaying { player.play() }
        case UInt8(ascii: "B"):
            // Bypass echo cancellation while Friday is silent: saves most of its CPU cost.
            let bypass = body.first == 1
            if input.isVoiceProcessingBypassed != bypass { input.isVoiceProcessingBypassed = bypass }
        case UInt8(ascii: "S"):
            player.stop()  // drops queued buffers (their completions still fire)
            player.play()
        case UInt8(ascii: "Q"):
            exit(0)
        default:
            break
        }
    }
}.start()

send(UInt8(ascii: "R"))
RunLoop.main.run()
