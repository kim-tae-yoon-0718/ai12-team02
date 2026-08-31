// kordoc으로 hwp/pdf를 markdown으로 변환하고 품질 신호를 함께 뽑는다.
//
// parse_file.py가 호출한다. 파일 하나당 JSON 한 줄을 stdout에 쓴다.
// kordoc CLI로는 qualitySummary(needsOcr 등)를 받을 수 없어 API를 직접 호출한다.
//
// 실행 (직접 부를 일은 거의 없다):
//   node kordoc_convert.mjs <file...> -d <outDir>
//
// ⚠️ kordoc 라이브러리는 이 파일과 같은 곳에 있지 않다.
//    node_modules는 공용 폴더에 있고, parse_file.py가 cwd를 그쪽으로 지정해 실행한다.

import { parse } from "kordoc"
import { writeFileSync } from "fs"
import { basename, extname, join } from "path"

const args = process.argv.slice(2)
const dIndex = args.indexOf("-d")
if (dIndex === -1) {
    console.error("Usage: node kordoc_convert.mjs <file...> -d <outDir>")
    process.exit(1)
}
const outDir = args[dIndex + 1]
const files = args.slice(0, dIndex)

for (const file of files) {
    const stem = basename(file, extname(file))
    let result

    try {
        result = await parse(file)
    } catch (err) {
        console.log(JSON.stringify({ file, success: false, stage: "parse", thrown: String(err) }))
        continue
    }

    if (!result.success) {
        console.log(JSON.stringify({ file, success: false, stage: "parse", result }))
        continue
    }

    // 쓰기 실패가 배치 전체를 죽이지 않게 한다.
    // 디스크·권한 문제로 한 파일이 실패해도 나머지는 계속 변환한다.
    try {
        writeFileSync(join(outDir, `${stem}.md`), result.markdown)
    } catch (err) {
        console.log(JSON.stringify({ file, success: false, stage: "write", thrown: String(err) }))
        continue
    }

    console.log(JSON.stringify({
        file,
        success: true,
        needsOcr: result.qualitySummary?.needsOcr ?? false,
        ocrCandidatePages: result.qualitySummary?.ocrCandidatePages ?? null,
    }))
}
