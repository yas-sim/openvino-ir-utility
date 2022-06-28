# Disassemble OpenVINO Compiled model data

import os, sys

def get_dword(barray, pos):
    val=0
    for i in range(3, -1, -1):
        val = val<<8 | barray[pos+i]
    return val

def extract_block(barray, start, size, text_flag=False):
    block = barray[start:start+size]
    if text_flag:
        block = block.decode('utf8')
    return block

def extract_block_to_file(barray, start, size, file_name, text_flag=False):
    block = extract_block(barray, start, size, text_flag)
    fmode = 'w' if text_flag else 'wb'
    with open(file_name, fmode) as f:
        f.write(block)
    print('Generated \'{}\' : Offset 0x{:08x} Size 0x{:08x}'.format(file_name, start, size))

def main():
    assert len(sys.argv)>1
    fn = sys.argv[1]
    assert os.path.isfile(fn)

    with open(fn, 'rb') as f:
        model = f.read()

    base_fn, ext = os.path.splitext(fn)

    # Check model sile signature
    ofst = get_dword(model, 0x00)
    signature = extract_block(model, ofst, 13, True)
    assert signature == '<?xml version'

    # Extract XML header
    ofst = get_dword(model, 0x00)
    size = get_dword(model, 0x08)
    extract_block_to_file(model, ofst, size, base_fn+'_hdr.xml', True)

    # Extract optimized weight
    ofst = get_dword(model, 0x10)
    size = get_dword(model, 0x18)
    extract_block_to_file(model, ofst, size, base_fn+'_.bin', False)

    # Extract optimized graph
    ofst = get_dword(model, 0x20)
    size = get_dword(model, 0x28)
    extract_block_to_file(model, ofst, size, base_fn+'_.xml', True)

if __name__ == '__main__':
    sys.exit(main())
