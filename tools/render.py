"""Render nav grid to PNG: white=walkable, gray=blocked, dark=building tall, red=cover>0."""
import sys, zlib, struct
sys.path.insert(0, '.')
import nav
def png(path, w, h, rows):
    raw = b''.join(b'\x00' + bytes(r) for r in rows)
    def chunk(t, d): return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    open(path, 'wb').write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))
m = nav.Map(sys.argv[1])
rows = []
for j in reversed(range(m.nny)):
    r = []
    for i in range(m.nnx):
        k = j * m.nnx + i
        if m.walk[k]:
            c = (255, 255, 255) if m.cover[k] == 0 else (255, 200, 200)
        else:
            x, y = m.ncenter(k); fk = m.fidx(x, y)
            c = (60, 60, 90) if fk >= 0 and m.ftop[fk] > m.fground[fk] + 3 else (150, 150, 150)
        r += c
    rows.append(r)
png(sys.argv[2], m.nnx, m.nny, rows)
