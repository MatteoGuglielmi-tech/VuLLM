# NOTE: import modules from source with relative path to avoid
# pytest to break
import pytest
from restructure.treesitter import TreeSitter

testdata_for_ts_c = [
    (
        "#if __KERNEL__==4 static inline struct ext4_sb_info *EXT4_SB(struct super_block *sb)",
        "static inline struct ext4_sb_info *EXT4_SB(struct super_block *sb)",
    ),
    (
        "static inline struct ext4_sb_info *EXT4_SB(struct super_block *sb) #if __KERNEL__==4",
        "static inline struct ext4_sb_info *EXT4_SB(struct super_block *sb)",
    ),
    (
        "#if __KERNEL__<=4 static inline struct ext4_sb_info *EXT4_SB(struct super_block *sb)",
        "static inline struct ext4_sb_info *EXT4_SB(struct super_block *sb)",
    ),
    (
        "static inline struct ext4_sb_info *EXT4_SB(struct super_block *sb) #if __KERNEL__>=4",
        "static inline struct ext4_sb_info *EXT4_SB(struct super_block *sb)",
    ),
]

testdata_for_ts_cpp = [
    (
        "#if cimg_verbosity>=3 CImg<T> *data(const unsigned int pos)",
        "CImg<T> *data(const unsigned int pos)",
    ),
    (
        "CImg<T> *data(const unsigned int pos) #if cimg_verbosity>=3",
        "CImg<T> *data(const unsigned int pos)",
    ),
    (
        "#if cimg_verbosity>3 CImg<T> *data(const unsigned int pos)",
        "CImg<T> *data(const unsigned int pos)",
    ),
    (
        "CImg<T> *data(const unsigned int pos) #if cimg_verbosity>3",
        "CImg<T> *data(const unsigned int pos)",
    ),
    (
        "#if cimg_verbosity<=3 T *data(const unsigned int x, const unsigned int y=0, const unsigned int z=0, const unsigned int c=0)",
        "T *data(const unsigned int x, const unsigned int y=0, const unsigned int z=0, const unsigned int c=0)",
    ),
    (
        "T *data(const unsigned int x, const unsigned int y=0, const unsigned int z=0, const unsigned int c=0) #if cimg_verbosity<=3",
        "T *data(const unsigned int x, const unsigned int y=0, const unsigned int z=0, const unsigned int c=0)",
    ),
]


@pytest.fixture
def c_ts():
    return TreeSitter(language_name="c")


@pytest.fixture
def cpp_ts():
    return TreeSitter(language_name="cpp")


@pytest.mark.parametrize("a,b", testdata_for_ts_c)
def test_remove_comparative_preprocessor_conditions_c(c_ts: TreeSitter, a: str, b: str):
    src: str = c_ts.remove_if_condition(a)
    assert src == b


@pytest.mark.parametrize("a,b", testdata_for_ts_cpp)
def test_remove_comparative_preprocessor_conditions_cpp(
    cpp_ts: TreeSitter, a: str, b: str
):
    src: str = cpp_ts.remove_if_condition(a)
    assert src == b
