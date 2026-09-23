/**
 * webui/js/sim/bundle.mjs — 读 `webui/data/` 数据包 (契约 contracts/webui_bundle_v1.md)。
 *
 * 纯计算 ES module: node 与浏览器共用, 不碰 DOM。IO 由调用方注入:
 *   浏览器  `httpIO('data/')`        (fetch)
 *   node    `await nodeIO('webui/data')` (fs)
 *
 * **浏览器内解压 (09-22)**: 静态托管 (GitHub Pages) 不会对 `.onnx` / `.STL` / `.f32` 做 gzip, 也不让改响应头,
 * 而首屏大头恰好全是这些。所以 `build.py` 额外产出 `<path>.gz` **旁车** (原文件保留), 这里先试旁车 + 就地
 * `DecompressionStream('gzip')` 解压, 拿不到才回落原文件。哪些文件有旁车看 `index.json` 的 `gz.files`
 * (`loadIndex` 会把它挂到 io 上), 所以不会为没有旁车的文件白跑一趟 404。
 * 进度回调报的是**解压后**字节 (与 index.json 里的 bytes 同口径); 真正下行的字节在 `io.stats.wire`。
 */

/** 浏览器有没有 DecompressionStream (Chrome 80+ / Edge 80+ / Firefox 113+ / Safari 16.4+; node 18+ 也有)。 */
export const GZIP_SUPPORTED = typeof DecompressionStream !== 'undefined';

/** gzip 字节 -> 原始字节 (流式, 不占两份大内存峰值以上)。 */
export async function gunzip(packed) {
  const ds = new DecompressionStream('gzip');
  const stream = new Blob([packed]).stream().pipeThrough(ds);
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/**
 * 取一个 URL, 优先取 `<url>.gz` 并就地解压。
 * @param url      原文件 URL
 * @param tryGz    true = 先试旁车 (由清单决定, 别对没有旁车的文件开)
 * @param init     可选 fetch init (只有 index.json 用: `{cache: 'no-cache'}` 强制回源校验, trap #35)
 * @returns {Promise<{bytes: Uint8Array, wire: number, gz: boolean}>}
 */
export async function fetchMaybeGz(url, tryGz = true, init = undefined) {
  if (tryGz && GZIP_SUPPORTED) {
    let r = null;
    try {
      r = await fetch(url + '.gz', init);
    } catch (e) {
      r = null;                                   // 网络层直接失败: 回落原文件 (别让旁车拖垮页面)
    }
    if (r && r.ok) {
      const packed = new Uint8Array(await r.arrayBuffer());
      // 有的服务器会对 .gz 再加一个 `Content-Encoding: gzip` (浏览器已经替我们解了一层):
      // 按 gzip 魔数判断, 别解第二次。
      const isGz = packed.length > 2 && packed[0] === 0x1f && packed[1] === 0x8b;
      return { bytes: isGz ? await gunzip(packed) : packed, wire: packed.byteLength, gz: isGz };
    }
  }
  const r2 = await fetch(url, init);
  if (!r2.ok) throw new Error(`${url}: HTTP ${r2.status}`);
  const b = new Uint8Array(await r2.arrayBuffer());
  // content-length = **传输**字节 (服务器自己 gzip 时比解出来的小); 同源才读得到, 读不到就按解压后算
  const cl = Number(r2.headers.get('content-length'));
  return { bytes: b, wire: Number.isFinite(cl) && cl > 0 ? cl : b.byteLength, gz: false };
}

/** @param onProgress 可选 `(bytes, path) => void`: 每个文件取完报一次**解压后**的字节数 (首屏进度条用) */
export function httpIO(base, onProgress = null) {
  const url = (p) => new URL(p, base).href;
  const io = {
    base,
    onProgress,
    gz: null,                                     // loadIndex() 填: index.json 的 gz 节 ({files: {...}})
    stats: { wire: 0, plain: 0, gz_hits: 0, files: 0 },
    /** 这个路径有没有 .gz 旁车 (清单说了算; 没有清单就谁都不试) */
    hasGz(p) { return !!(this.gz && this.gz.files && this.gz.files[p]); },
    async bytes(p, init) {
      const r = await fetchMaybeGz(url(p), this.hasGz(p), init);
      this.stats.wire += r.wire;
      this.stats.plain += r.bytes.byteLength;
      this.stats.files += 1;
      if (r.gz) this.stats.gz_hits += 1;
      if (this.onProgress) this.onProgress(r.bytes.byteLength, p);
      return r.bytes;
    },
    async text(p, init) { return new TextDecoder().decode(await this.bytes(p, init)); },
    async json(p, init) { return JSON.parse(await this.text(p, init)); },
  };
  return io;
}

export async function nodeIO(root) {
  const fs = await import('node:fs/promises');
  const path = await import('node:path');
  return {
    base: root,
    gz: null,
    async bytes(p) { return new Uint8Array(await fs.readFile(path.join(root, p))); },
    async text(p) { return await fs.readFile(path.join(root, p), 'utf8'); },
    async json(p) { return JSON.parse(await this.text(p)); },
  };
}

/** index.json (策略 / 地形 / 机器人 / 版本 / gz 旁车清单)。 */
export async function loadIndex(io) {
  // **`index.json` 必须回源校验** (trap #35): 它是数据包的版本指针, 其余文件都按它的清单取。
  // GitHub Pages 发 `cache-control: max-age=600`, 老访客的浏览器缓存会让新推的包整个看不见
  // (09-23 实撞: 站上是 5 个策略, 页面里还是 09-22 的 3 个)。`no-cache` = 带 ETag 条件请求,
  // 命中就是一个 304, 9 KB 的东西不值得为它省一次 RTT。nodeIO 的 json() 忽略这个参数。
  const index = await io.json('index.json', { cache: 'no-cache' });
  // 旁车清单挂到 io 上 -> 之后所有 io.bytes() 自动走 .gz (node 侧的 nodeIO 直接读盘, 用不上)
  if (index && index.gz && index.gz.files) io.gz = index.gz;
  return index;
}

/** 机器人: 预处理过的 MJCF 文本 + 网格字节 (VFS 用)。 */
export async function loadRobot(io, index) {
  const xml = await io.text(index.robot.xml);
  const meshes = new Map();
  await Promise.all(index.robot.meshes.map(async (f) => {
    meshes.set(f, await io.bytes(`${index.robot.meshdir}/${f}`));
  }));
  return { xml, meshes };
}

/** 地形: <id>.json 元数据 (+ hfield 的话再取 <id>.f32 绝对高度; plane 没有 .f32)。 */
export async function loadTerrainData(io, id) {
  const meta = await io.json(`terrains/${id}.json`);
  if ((meta.repr ?? 'hfield') !== 'hfield') return { meta, heights: null };
  const raw = await io.bytes(`terrains/${meta.hfield.data}`);
  // 解压出来的 Uint8Array 是新开的 buffer, byteOffset 为 0; 但原文件回落那条路可能带偏移, 两条都照顾到。
  const aligned = (raw.byteOffset % 4 === 0) ? raw : new Uint8Array(raw);
  const heights = new Float32Array(aligned.buffer, aligned.byteOffset, aligned.byteLength / 4);
  if (heights.length !== meta.nrow * meta.ncol) throw new Error(`terrain ${id}: f32 长度 ${heights.length} != nrow*ncol ${meta.nrow * meta.ncol}`);
  return { meta, heights };
}

/** 策略: meta.json + contract.json + ONNX 字节 (按 meta.files)。 */
export async function loadPolicyBundle(io, name) {
  const dir = `policies/${name}`;
  const meta = await io.json(`${dir}/meta.json`);
  const contract = await io.json(`${dir}/contract.json`);
  const onnx = {};
  for (const [k, f] of Object.entries(meta.files)) onnx[k] = await io.bytes(`${dir}/${f}`);
  return { meta, contract, onnx };
}
