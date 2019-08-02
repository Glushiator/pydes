# -*- coding: utf8 -*-

from __future__ import print_function

from misc_tools import TimeIt, recursive_map, flatten, memoize, memoize_range


@memoize_range(16)
def int2_4bits(val):
    """return the binary value as a tuple of 4 bits"""
    binval = bin(val)[2:]
    if len(binval) > 4:
        raise DESError("binary value larger than the expected size")
    result = tuple([int(x) for x in binval.rjust(4, '0')])
    return result


@memoize_range(256)
def int2_8bits(val):
    """return the binary value as a tuple of 8 bits"""
    binval = bin(val)[2:]
    if len(binval) > 8:
        raise DESError("binary value larger than the expected size")
    result = tuple([int(x) for x in binval.rjust(8, '0')])
    return result


@memoize
def bits2char(bits):
    return chr(int(''.join(map(str, bits)), 2))


class DESError(Exception):
    pass


def permutation_table(*table):
    """takes a permutation table and shifts indices down by 1
    permutation tables are given a arrays of indices in range [1, N]
    but arrays being permuted have indices in range [0, N-1]"""
    return tuple(recursive_map(lambda x: x - 1, table))


# Initial permut matrix for the datas
PI = permutation_table(
     58, 50, 42, 34, 26, 18, 10, 2,
     60, 52, 44, 36, 28, 20, 12, 4,
     62, 54, 46, 38, 30, 22, 14, 6,
     64, 56, 48, 40, 32, 24, 16, 8,
     57, 49, 41, 33, 25, 17, 9, 1,
     59, 51, 43, 35, 27, 19, 11, 3,
     61, 53, 45, 37, 29, 21, 13, 5,
     63, 55, 47, 39, 31, 23, 15, 7)

# Initial permut made on the key
CP_1 = permutation_table(
     57, 49, 41, 33, 25, 17, 9,
     1, 58, 50, 42, 34, 26, 18,
     10, 2, 59, 51, 43, 35, 27,
     19, 11, 3, 60, 52, 44, 36,
     63, 55, 47, 39, 31, 23, 15,
     7, 62, 54, 46, 38, 30, 22,
     14, 6, 61, 53, 45, 37, 29,
     21, 13, 5, 28, 20, 12, 4)

# Permut applied on shifted key to get Ki+1
CP_2 = permutation_table(
     14, 17, 11, 24, 1, 5, 3, 28,
     15, 6, 21, 10, 23, 19, 12, 4,
     26, 8, 16, 7, 27, 20, 13, 2,
     41, 52, 31, 37, 47, 55, 30, 40,
     51, 45, 33, 48, 44, 49, 39, 56,
     34, 53, 46, 42, 50, 36, 29, 32)

# Expand matrix to get a 48bits matrix of datas to apply the xor with Ki
E = permutation_table(
     32, 1, 2, 3, 4, 5,
     4, 5, 6, 7, 8, 9,
     8, 9, 10, 11, 12, 13,
     12, 13, 14, 15, 16, 17,
     16, 17, 18, 19, 20, 21,
     20, 21, 22, 23, 24, 25,
     24, 25, 26, 27, 28, 29,
     28, 29, 30, 31, 32, 1)


S_BOX = tuple(recursive_map(int2_4bits, (

    flatten
    ((14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7),
     (0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8),
     (4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0),
     (15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13),
     ),

    flatten
    ((15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10),
     (3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5),
     (0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15),
     (13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9),
     ),

    flatten
    ((10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8),
     (13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1),
     (13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7),
     (1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12),
     ),

    flatten
    ((7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15),
     (13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9),
     (10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4),
     (3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14),
     ),

    flatten
    ((2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9),
     (14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6),
     (4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14),
     (11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3),
     ),

    flatten
    ((12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11),
     (10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8),
     (9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6),
     (4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13),
     ),

    flatten
    ((4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1),
     (13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6),
     (1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2),
     (6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12),
     ),

    flatten
    ((13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7),
     (1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2),
     (7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8),
     (2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11),
     )
)))

# Permut made after each SBox substitution for each round
P = permutation_table(
     16, 7, 20, 21, 29, 12, 28, 17,
     1, 15, 23, 26, 5, 18, 31, 10,
     2, 8, 24, 14, 32, 27, 3, 9,
     19, 13, 30, 6, 22, 11, 4, 25)

# Final permut for datas after the 16 rounds
PI_1 = permutation_table(
     40, 8, 48, 16, 56, 24, 64, 32,
     39, 7, 47, 15, 55, 23, 63, 31,
     38, 6, 46, 14, 54, 22, 62, 30,
     37, 5, 45, 13, 53, 21, 61, 29,
     36, 4, 44, 12, 52, 20, 60, 28,
     35, 3, 43, 11, 51, 19, 59, 27,
     34, 2, 42, 10, 50, 18, 58, 26,
     33, 1, 41, 9, 49, 17, 57, 25)

# Matrix that determine the shift for each of 16 rounds of keys
SHIFT = (1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1)


def string_to_bit_array(text):
    """Convert a string into a list of bits"""
    array = []
    for char in text:
        # Add the bits to the final list
        array.extend(int2_8bits(ord(char)))
    return tuple(array)


def bit_array_to_string(array):
    """Recreate the string from the bit array"""
    res = ''.join((bits2char(tuple(octet)) for octet in nsplit8(array)))
    return res


def add_padding(text):
    """Add padding to the text using PKCS5 spec."""
    pad_len = 8 - (len(text) % 8)
    return text + pad_len * chr(pad_len)


def remove_padding(data):
    """Remove the padding of the plain text (it assumes there is padding)"""
    pad_len = ord(data[-1])
    return data[:-pad_len]


def nsplit(s, n):
    """Split a list into sublists of size "n", returns a generator."""
    return (s[k: k + n] for k in range(0, len(s), n))


def nsplit6(s):
    """Split a list into sublists of size "6", returns a generator."""
    return (s[k: k + 6] for k in range(0, len(s), 6))


def nsplit8(s):
    """Split a list into sublists of size "8", returns a generator."""
    return (s[k: k + 8] for k in range(0, len(s), 8))


def shift(g, d, n):
    """Shift a list of the given value"""
    return g[n:] + g[:n], d[n:] + d[:n]


def xor(t1, t2):
    """Apply a xor and return the resulting list"""
    return [x ^ y for x, y in zip(t1, t2)]


def expand(block, table):
    """Do the exact same thing than permut but for more clarity has been renamed"""
    return [block[x] for x in table]


def permut(block, table):
    """Permut the given block using the given table (so generic method)"""
    return [block[x] for x in table]


def substitute(d_e):
    """Substitute bytes using SBOX"""
    result = []
    # Split bit array into sublist of 6 bits
    for i, subblock in enumerate(nsplit6(d_e)):  # For all the sublists
        # unpack a block into bits
        b1, b2, b3, b4, b5, b6 = subblock
        # Get the row with the first and last bit
        # ## row = (b1 << 1) + b6
        # Column is the 2, 3, 4, 5th bits
        # ## column = (b2 << 3) + (b3 << 2) + (b4 << 1) + b5
        # Take the value in the SBOX appropriated for the round (i)
        # And append it to the resulting list
        row_column = (b1 << 5) | (b6 << 4) | (b2 << 3) | (b3 << 2) | (b4 << 1) | b5
        result.extend(S_BOX[i][row_column])
    return result


class DES(object):

    def __init__(self, password, padding=False):
        if len(password) < 8:
            raise DESError("password must be at least 8 bytes long, got {}".format(len(password)))
        elif len(password) > 8:
            # If key size is above 8 bytes, cut to be 8bytes long
            password = password[:8]
        self.padding = padding
        # Generate all the keys
        self.encryption_keys = self.generate_keys(password)
        self.decryption_keys = self.encryption_keys[::-1]

    @staticmethod
    def run(keys, text):
        processed = []
        # Split the text in blocks of 8 bytes so 64 bits
        # Loop over all the blocks of data
        for block in nsplit8(text):
            block = string_to_bit_array(block)  # Convert the block in bit array
            block = permut(block, PI)  # Apply the initial permutation
            g, d = nsplit(block, 32)  # g(LEFT), d(RIGHT)
            # Do the 16 rounds
            for key in keys:
                # Expand d to match Ki size (48bits)
                d_e = expand(d, E)
                tmp = xor(key, d_e)
                # Method that will apply the SBOXes
                tmp = substitute(tmp)
                tmp = permut(tmp, P)
                tmp = xor(g, tmp)
                g = d
                d = tmp
            # Do the last permut and append to the processed blocks
            processed.append(bit_array_to_string(permut(d + g, PI_1)))
        return ''.join(processed)

    @staticmethod
    def generate_keys(password):
        """Algorithm that generates all the keys"""
        keys = []
        key = string_to_bit_array(password)
        # Apply the initial permut on the key
        key = permut(key, CP_1)
        # Split it in to (g->LEFT),(d->RIGHT)
        g, d = nsplit(key, 28)
        # Apply the 16 rounds
        for shift_ in SHIFT:
            # Apply the shift associated with the round (not always 1)
            g, d = shift(g, d, shift_)
            # Merge them
            tmp = g + d
            # Apply the permut to get the Ki
            a_key = permut(tmp, CP_2)
            keys.append(tuple(a_key))
        return tuple(keys)

    def encrypt(self, text):
        if self.padding:
            text = add_padding(text)
        elif len(text) % 8 != 0:  # If not padding specified data size must be multiple of 8 bytes
            raise DESError("text size should be multiple of 8, got {}".format(len(text)))
        return self.run(self.encryption_keys, text)

    def decrypt(self, text):
        result = self.run(self.decryption_keys, text)
        if self.padding:
            # Remove the padding if decrypt and padding is true
            result = remove_padding(result)
        # Return the final string of data ciphered/deciphered
        return result


def main():
    from binascii import hexlify
    from binascii import unhexlify
    import os
    key = "secret_k"
    text = "12345678"
    d = DES(key)
    r = d.encrypt(text)
    r2 = d.decrypt(r)
    assert text == r2
    # check_equal('\xdf\xe5\xfd\xe5\xda\x9f\\\x9d\x86j\xdb\xfa\xdd.\xe2\x10', r)
    text = "0123456701234567"
    d = DES(key)
    r = d.encrypt(text)
    assert '70dec76f5a7d925b70dec76f5a7d925b' == hexlify(r)
    test_vectors = [
        ('0000000000000000', '3b3898371520f75e', '83a1e814889253e0'),
        ('83a1e814889253e0', '3b3898371520f75e', '5ff9376e64834f21'),
        ('5ff9376e64834f21', '3b3898371520f75e', '343c39c4d5589c42'),
    ]
    for n, (clear, key, expected) in enumerate(test_vectors):
        test_des = DES(unhexlify(key))
        calculated = hexlify(test_des.encrypt(unhexlify(clear)))
        print(n, ":", calculated, expected, calculated == expected)
    sample_data = os.urandom(64 * 1024)
    sample_key = os.urandom(8)
    cipher = DES(sample_key)
    timer = TimeIt("speed test: encryption of {} bytes".format(len(sample_data)))
    with timer:
        encryped = cipher.encrypt(sample_data)
    print("{:.2f} kB/sec".format(len(sample_data) / 1024 / timer.elapsed))
    timer = TimeIt("speed test: decryption of {} bytes".format(len(sample_data)))
    with timer:
        decrypted = cipher.decrypt(encryped)
    assert sample_data == decrypted
    print("{:.2f} kB/sec".format(len(sample_data) / 1024 / timer.elapsed))


if __name__ == '__main__':
    main()
