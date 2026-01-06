import heapq
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, namedtuple
from PIL import Image


class Node(namedtuple("Node", ["value", "freq", "left", "right"])):
    def __lt__(self, other):
        return self.freq < other.freq


def make_frequency_dict(data):
    return Counter(data)


def build_huffman_tree(frequency):
    heap = []
    for value in frequency:
        node = Node(value, frequency[value], None, None)
        heapq.heappush(heap, node)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(None, left.freq + right.freq, left, right)
        heapq.heappush(heap, merged)

    return heapq.heappop(heap)


def generate_codes(root):
    codes = {}
    reverse_mapping = {}

    def make_codes_helper(node, current_code):
        if node is None:
            return
        if node.value is not None:
            codes[node.value] = current_code
            reverse_mapping[current_code] = node.value
            return
        make_codes_helper(node.left, current_code + "0")
        make_codes_helper(node.right, current_code + "1")

    make_codes_helper(root, "")
    return codes, reverse_mapping


def encode_data(data, codes):
    encoded_text = ""
    for item in data:
        encoded_text += codes[item]
    return encoded_text


def pad_encoded_data(encoded_text):
    extra_padding = 8 - len(encoded_text) % 8
    encoded_text += "0" * extra_padding
    padded_info = "{0:08b}".format(extra_padding)
    return padded_info + encoded_text


def get_byte_array(padded_encoded_text):
    b = bytearray()
    for i in range(0, len(padded_encoded_text), 8):
        byte = padded_encoded_text[i : i + 8]
        b.append(int(byte, 2))
    return b


def remove_padding(padded_encoded_data):
    padded_info = padded_encoded_data[:8]
    extra_padding = int(padded_info, 2)
    encoded_text = padded_encoded_data[8:]
    return encoded_text[:-extra_padding]


def decode_data(encoded_text, reverse_mapping):
    current_code = ""
    decoded_data = []
    for bit in encoded_text:
        current_code += bit
        if current_code in reverse_mapping:
            decoded_data.append(reverse_mapping[current_code])
            current_code = ""
    return decoded_data


def huffman_compress(image_array):
    flat_data = image_array.flatten()
    frequency = make_frequency_dict(flat_data)
    root = build_huffman_tree(frequency)
    codes, reverse_mapping = generate_codes(root)

    encoded_data = encode_data(flat_data, codes)
    padded_encoded_data = pad_encoded_data(encoded_data)
    compressed_data = get_byte_array(padded_encoded_data)

    return (
        compressed_data,
        codes,
        reverse_mapping,
        len(flat_data) * 8,
        len(padded_encoded_data),
    )


def huffman_decompress(compressed_data, reverse_mapping, original_shape):
    bit_string = ""
    for byte in compressed_data:
        bits = bin(byte)[2:].rjust(8, "0")
        bit_string += bits

    encoded_text = remove_padding(bit_string)
    decoded_data = decode_data(encoded_text, reverse_mapping)
    return np.array(decoded_data, dtype=np.uint8).reshape(original_shape)


image_path = "/home/hasan/Desktop/Hasan/7th Semester/CSE4161_2_DIP/Code/original.png"
img = Image.open(image_path).convert("L")
image_array = np.array(img)

# Compress the image
compressed_data, codes, reverse_mapping, original_bits, compressed_bits = (
    huffman_compress(image_array)
)

# Decompress the image
decompressed_image = huffman_decompress(
    compressed_data, reverse_mapping, image_array.shape
)

# Calculate compression ratio
compression_ratio = original_bits / compressed_bits
print(f"Original size: {original_bits} bits")
print(f"Compressed size: {compressed_bits} bits")
print(f"Compression Ratio: {compression_ratio:.2f}")

# Display results
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image_array, cmap="gray")
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(decompressed_image, cmap="gray")
axes[1].set_title("Reconstructed Image")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("images/output/huffman_result.png", dpi=300)
plt.show()