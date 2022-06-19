#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cassert>

using namespace std;
namespace py = pybind11;
struct ExactCover {
    vector<vector<int>> rows;
    unordered_map<int, unordered_set<int>> cols;
    vector<bool> already_used;
    int n, m;
    bool prepared = false;

    ExactCover(int _n, int _m) : n(_n), m(_m), rows(_n), already_used(_n) {}

    void add_element_to_way(int way, int element) {
        assert(way < n);
        assert(element < m);
        cols[element].insert(way);
        rows[way].push_back(element);
    }

    void set_already_used(int way) {
        already_used[way] = 1;
    }

    void prepare() {
        prepared = 1;
        for (auto &e: rows)sort(e.begin(), e.end());
        for (int i = 0; i < n; i++)if (already_used[i]) extract_intersects(i);
    }

    vector<unordered_set<int>> extract_intersects(int row_index) {
        vector<unordered_set<int>> buf;
        for (auto &col: rows[row_index]) {
            buf.push_back(cols[col]);
            cols.erase(col);
            for (auto &interesting_row: buf.back()) {
                for (auto &other_col: rows[interesting_row]) {
                    if (other_col != col) {
                        cols[other_col].erase(interesting_row);
                    }
                }
            }
        }
        return buf;
    }

    void restore_intersects(int row_index, vector<unordered_set<int>> &buf) {
        for (auto it = rows[row_index].rbegin(); it != rows[row_index].rend(); ++it) {
            auto col = *it;
            cols[col] = buf.back();
            buf.pop_back();
            for (auto &added_row: cols[col]) {
                for (auto &other_col: rows[added_row]) {
                    cols[other_col].insert(added_row);
                }
            }
        }
    }

    bool _get_exact_cover(vector<int> &result) {
        if (cols.empty()) return true;
        int mi = cols.begin()->first;
        for (auto &e: cols) {
            if (e.second.size() < cols[mi].size()) mi = e.first;
        }
        auto cols_copy = cols[mi];
        for (auto row: cols_copy) {
            result.push_back(row);
            auto buf = extract_intersects(row);
            auto b = _get_exact_cover(result);
            if (b) return true;
            restore_intersects(row, buf);
            result.pop_back();
        }
        return false;
    }

    // find one solution if it exists
    bool get_exact_cover(vector<int> &result) {
        if (!prepared) {
            throw std::runtime_error("Exact matching was not prepared. Call prepare");
        }
        unordered_set<int> temp;
        for (int i = 0; i < n; i++) {
            for (auto &e: rows[i]) temp.insert(e);
        }
        if (temp.size() != m) return false;
        return _get_exact_cover(result);
    }
};

class SudokuDeck {
    int BLOCK_SIZE;
    int n;
    vector<vector<int>> deck;
public:
    SudokuDeck(int block_size) : BLOCK_SIZE(block_size), n(block_size * block_size), deck(n, vector<int>(n, 0)) {}

    void set(int i, int j, int val) {
        deck[i][j] = val;
    }

    int get(int i, int j) {
        return deck[i][j];
    }
    vector<vector<int>> get_deck(){
        return deck;
    }
    bool solve() {
        // rows - (num, row ,col)
        /*
        (0, row, col) - на пересечении row и col стоит число
        (1, row, num) - в строке row есть число num
        (2, col, num) - в столбце col есть число num
        (3, q, num) - в квадранте q есть число num*/
        ExactCover cover(n * n * n, 4 * n * n);
        for(int num = 0; num < n; num++){
            for(int row = 0; row < n; row ++){
                for(int col = 0 ; col < n; col++){
                    int q = (row / BLOCK_SIZE) * BLOCK_SIZE + col / BLOCK_SIZE;
                    cover.add_element_to_way(get_index(num,row,col), get_index(0,row,col));
                    cover.add_element_to_way(get_index(num,row,col), get_index(1,row,num));
                    cover.add_element_to_way(get_index(num,row,col), get_index(2,col,num));
                    cover.add_element_to_way(get_index(num,row,col), get_index(3,q,num));
                }
            }
        }
        for(int i =0; i < n; i ++){
            for(int j =0 ; j < n; j ++){
                if(get(i,j)){
                    cover.set_already_used(get_index(get(i,j)-1,i,j));
                }
            }
        }
        cover.prepare();
        vector<int> res;
        bool ok = cover.get_exact_cover(res);
        if(!ok) return ok;
        for(auto &e : res){
            auto ind = get_rev_index(e);
            set(ind[1], ind[2], ind[0]+1);
        }
        return check();
    }
    bool check(){
        int c = 0;
        for (auto &e: deck) c += count(e.begin(), e.end(), 0);
        if (c)return 0;
        // check rows
        for (int i = 0; i < n; i++) {
            vector<bool> used(n);
            for (auto &e: deck[i]) used[e - 1] = 1;
            if (std::any_of(used.begin(), used.end(), [](bool b) -> bool { return !b; })) return false;
        }
        // check cols
        for (int i = 0; i < n; i++) {
            vector<bool> used(n);
            for (int j = 0; j < n; j++)
                used[get(j, i) - 1] = 1;
            if (std::any_of(used.begin(), used.end(), [](bool b) -> bool { return !b; })) return false;
        }
        for (int i = 0; i < BLOCK_SIZE; i++) {
            for (int j = 0; j < BLOCK_SIZE; j++) {
                vector<bool> used(n);
                for (int k1 = i * BLOCK_SIZE; k1 < i * BLOCK_SIZE + BLOCK_SIZE; k1++) {
                    for (int k2 = j * BLOCK_SIZE; k2 < j * BLOCK_SIZE + BLOCK_SIZE; k2++) {
                        used[get(k1, k2) - 1] = 1;
                    }
                }
                if (std::any_of(used.begin(), used.end(), [](bool b) -> bool { return !b; })) return false;
            }
        }
        return true;
    }
    friend ostream &operator<<(ostream &fos, SudokuDeck &deck);
private:
    int get_index(int i, int j, int k) {
        return i * n * n + j * n + k;
    }
    vector<int> get_rev_index(int i){
        vector<int> res(3);
        res[2] = i % n;
        i -= res[2];
        i /= n;
        res[1] = i % n;
        i -= res[1];
        res[0] = i / n;
        return res;
    }
};

ostream &operator<<(ostream &fos, SudokuDeck &deck) {
    for (int i = 0; i < deck.n; i++) {
        for (int j = 0; j < deck.n; j++) {
            fos << deck.get(i, j) << " ";
            if ((j + 1) % deck.BLOCK_SIZE == 0 && j != deck.n - 1) {
                fos << "| ";
            }
        }
        fos << "\n";
        if ((i + 1) % deck.BLOCK_SIZE == 0 && i != deck.n - 1) {
            for (int j = 0; j < deck.n * 2 - 1 + 2 * (deck.BLOCK_SIZE - 1); j++)fos << '-';
            fos << "\n";
        }
    }
    return fos;
}

PYBIND11_MODULE(sudoku_solver, handle){
    handle.doc() = "CPP code for Knuth's DLX solving sudoku";
    py::class_<SudokuDeck>(handle,"SudokuDeck")
            .def(py::init<int>())
            .def("solve",&SudokuDeck::solve)
            .def("set",&SudokuDeck::set)
            .def("get",&SudokuDeck::get)
            .def("get_deck",&SudokuDeck::get_deck);
}
/*
 6 7
3 1 4 7
2 1 4
3 4 5 7
3 3 5 6
4 2 3 6 7
2 2 7
 * */
/*
 *     0 0 0 0 0 0 4 0 0
       3 0 6 0 0 0 0 0 0
       0 0 0 1 9 6 0 3 0
       0 7 0 0 0 0 0 1 0
       8 0 0 2 5 0 0 9 0
       0 4 0 0 0 0 8 0 0
       0 6 0 4 0 9 0 0 8
       0 0 5 0 0 0 0 2 0
       0 0 0 5 0 0 0 0 7
 * */