#include "MpsReader.hpp"

#include <cctype>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace sihps {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

std::vector<std::string> tokenize(const std::string& line) {
    std::vector<std::string> tokens;
    std::istringstream iss(line);
    std::string tok;
    while (iss >> tok) {
        tokens.push_back(tok);
    }
    return tokens;
}

// A handful of classic Netlib files (dfl001, forplan, gfrd-pnc, sierra,
// blend) rely on strict fixed-column MPS layout: row/column names may
// contain embedded blanks (e.g. "BR   1 1"), and the optional vector-name
// field on RHS/RANGES/BOUNDS lines may be blank rather than absent. Both
// break whitespace tokenization, which cannot tell an embedded blank in a
// name from a field separator. Fields are extracted at the standard fixed
// columns (1-indexed 2-3, 5-12, 15-22, 25-36, 40-47, 50-61) and used only as
// a fallback when free-format parsing of the whole file fails.
struct FixedFields {
    std::string f1, f2, f3, f4, f5, f6;
};

std::string fixed_field(const std::string& line, std::size_t start, std::size_t len) {
    if (start >= line.size()) return "";
    std::string field = line.substr(start, std::min(len, line.size() - start));
    std::size_t b = field.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    std::size_t e = field.find_last_not_of(" \t\r\n");
    return field.substr(b, e - b + 1);
}

FixedFields tokenize_fixed(const std::string& line) {
    FixedFields f;
    f.f1 = fixed_field(line, 1, 2);
    f.f2 = fixed_field(line, 4, 8);
    f.f3 = fixed_field(line, 14, 8);
    f.f4 = fixed_field(line, 24, 12);
    f.f5 = fixed_field(line, 39, 8);
    f.f6 = fixed_field(line, 49, 12);
    return f;
}

std::string normalized_token(std::string token) {
    if (token.size() >= 2 && token.front() == '\'' && token.back() == '\'') {
        token = token.substr(1, token.size() - 2);
    }
    for (char& c : token) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return token;
}

bool is_section_header(const std::string& line, const std::string& first_token) {
    if (line.empty() || line[0] == ' ' || line[0] == '\t') {
        return false; // data lines are indented in standard MPS layout
    }
    static const std::unordered_set<std::string> keywords = {
        "NAME", "ROWS", "COLUMNS", "RHS", "RANGES", "BOUNDS", "ENDATA", "OBJSENSE"};
    return keywords.count(first_token) > 0;
}

} // namespace

// `fixed_format` selects between whitespace tokenization (the common case)
// and strict fixed-column field extraction (the fallback for files whose
// row/column names contain embedded blanks, or that leave the optional
// RHS/RANGES/BOUNDS vector-name field blank). Both produce the same
// (name, value) pairs per data line; only how those pairs are extracted
// from the raw line differs.
MpsModel parse_mps_lines(const std::vector<std::string>& lines, bool fixed_format) {
    MpsModel model;
    std::unordered_map<std::string, std::int32_t> row_index; // L/G/E rows only
    std::unordered_set<std::string> declared_rows;            // every row, including extra N rows
    std::unordered_map<std::string, std::int32_t> col_index;
    std::vector<bool> has_explicit_lower; // tracks LO/FX/FR/MI/BV, for the negative-UP convention
    std::string section;
    bool integer_section = false;

    auto parse_double = [](const std::string& s, const std::string& what) -> double {
        try {
            std::size_t consumed = 0;
            double v = std::stod(s, &consumed);
            if (consumed != s.size()) throw std::invalid_argument("trailing characters");
            return v;
        } catch (const std::exception&) {
            throw std::runtime_error("MpsReader: expected a numeric " + what + ", got '" + s +
                                      "'");
        }
    };

    for (const std::string& line : lines) {
        if (line.empty() || line[0] == '*') {
            continue;
        }
        auto tokens = tokenize(line);
        if (tokens.empty()) {
            continue;
        }

        if (is_section_header(line, tokens[0])) {
            section = tokens[0];
            if (section == "NAME" && tokens.size() > 1) {
                model.name = tokens[1];
            }
            continue;
        }

        if (section == "ROWS") {
            std::string type, name;
            if (fixed_format) {
                FixedFields f = tokenize_fixed(line);
                type = f.f1;
                name = f.f2;
            } else {
                if (tokens.size() < 2) {
                    throw std::runtime_error("MpsReader: malformed ROWS line: " + line);
                }
                type = tokens[0];
                name = tokens[1];
            }
            if (name.empty()) {
                throw std::runtime_error("MpsReader: malformed ROWS line: " + line);
            }
            declared_rows.insert(name);

            if (type == "N") {
                if (model.objective_row_name.empty()) {
                    model.objective_row_name = name;
                }
                continue; // objective row, or a subsequent dropped free row
            }
            if (type.size() != 1 || (type[0] != 'L' && type[0] != 'G' && type[0] != 'E')) {
                throw std::runtime_error("MpsReader: unknown row type '" + type + "' for row " +
                                          name);
            }
            row_index[name] = model.n_rows++;
            model.row_names.push_back(name);
            model.row_types.push_back(type[0]);
            model.row_range.push_back(0.0);
            model.has_range.push_back(false);

        } else if (section == "COLUMNS") {
            // Free-format MPS integer markers are commonly written as
            // `MARK0000 'MARKER' 'INTORG'` and `MARK0001 'MARKER'
            // 'INTEND'. They are records, not columns. None of the files
            // that need the fixed-column fallback use markers, so this
            // check is safe to run unconditionally on whitespace tokens.
            if (tokens.size() >= 3 && normalized_token(tokens[1]) == "MARKER") {
                const std::string marker = normalized_token(tokens[2]);
                if (marker == "INTORG") {
                    integer_section = true;
                } else if (marker == "INTEND") {
                    integer_section = false;
                } else {
                    throw std::runtime_error("MpsReader: unknown COLUMNS marker: " + line);
                }
                continue;
            }

            std::string col_name;
            std::vector<std::pair<std::string, std::string>> pairs;
            if (fixed_format) {
                FixedFields f = tokenize_fixed(line);
                col_name = f.f2;
                if (!f.f3.empty()) pairs.push_back({f.f3, f.f4});
                if (!f.f5.empty()) pairs.push_back({f.f5, f.f6});
            } else {
                if (tokens.size() < 3) {
                    throw std::runtime_error("MpsReader: malformed COLUMNS line: " + line);
                }
                col_name = tokens[0];
                for (std::size_t k = 1; k + 1 < tokens.size(); k += 2) {
                    pairs.push_back({tokens[k], tokens[k + 1]});
                }
            }
            if (col_name.empty()) {
                throw std::runtime_error("MpsReader: malformed COLUMNS line: " + line);
            }

            std::int32_t c;
            auto cit = col_index.find(col_name);
            if (cit == col_index.end()) {
                c = model.n_cols++;
                col_index[col_name] = c;
                model.col_names.push_back(col_name);
                model.obj.push_back(0.0);
                model.col_lower.push_back(0.0);
                model.col_upper.push_back(kInf);
                model.col_types.push_back(integer_section ? VariableType::INTEGER
                                                           : VariableType::CONTINUOUS);
                has_explicit_lower.push_back(false);
            } else {
                c = cit->second;
                if (integer_section &&
                    model.col_types[static_cast<std::size_t>(c)] == VariableType::CONTINUOUS) {
                    model.col_types[static_cast<std::size_t>(c)] = VariableType::INTEGER;
                }
            }

            for (const auto& pr : pairs) {
                const std::string& row_name = pr.first;
                if (row_name == model.objective_row_name) {
                    model.obj[static_cast<std::size_t>(c)] = parse_double(pr.second, "COLUMNS value");
                    continue;
                }
                auto rit = row_index.find(row_name);
                if (rit != row_index.end()) {
                    model.constraint_triplets.push_back(
                        {rit->second, c, parse_double(pr.second, "COLUMNS value")});
                    continue;
                }
                if (declared_rows.count(row_name)) {
                    continue; // a legitimately-dropped extra free (N) row
                }
                throw std::runtime_error("MpsReader: COLUMNS references undeclared row: " +
                                          row_name);
            }

        } else if (section == "RHS" || section == "RANGES") {
            std::vector<std::pair<std::string, std::string>> pairs;
            if (fixed_format) {
                FixedFields f = tokenize_fixed(line);
                if (!f.f3.empty()) pairs.push_back({f.f3, f.f4});
                if (!f.f5.empty()) pairs.push_back({f.f5, f.f6});
            } else {
                // The vector-name field (field 2) is skipped unconditionally
                // here because every currently-passing free-format file
                // supplies it; files that leave it blank fail this pass and
                // are re-parsed in fixed_format mode instead.
                for (std::size_t k = 1; k + 1 < tokens.size(); k += 2) {
                    pairs.push_back({tokens[k], tokens[k + 1]});
                }
            }

            if (section == "RHS" && model.rhs.empty() && model.n_rows > 0) {
                model.rhs.assign(static_cast<std::size_t>(model.n_rows), 0.0);
            }

            for (const auto& pr : pairs) {
                const std::string& row_name = pr.first;
                if (row_name == model.objective_row_name) {
                    continue; // objective constant shift; not used by anything yet
                }
                auto rit = row_index.find(row_name);
                if (rit == row_index.end()) {
                    if (declared_rows.count(row_name)) continue; // dropped free row
                    throw std::runtime_error("MpsReader: " + section +
                                              " references undeclared row: " + row_name);
                }
                const double value = parse_double(pr.second, section + " value");
                if (section == "RHS") {
                    model.rhs[static_cast<std::size_t>(rit->second)] = value;
                } else {
                    model.row_range[static_cast<std::size_t>(rit->second)] = value;
                    model.has_range[static_cast<std::size_t>(rit->second)] = true;
                }
            }

        } else if (section == "OBJSENSE") {
            const std::string sense = normalized_token(tokens[0]);
            if (sense == "MIN") {
                model.objective_sense = ObjectiveSense::MINIMIZE;
            } else if (sense == "MAX") {
                model.objective_sense = ObjectiveSense::MAXIMIZE;
            } else {
                throw std::runtime_error("MpsReader: unknown objective sense: " + line);
            }

        } else if (section == "BOUNDS") {
            std::string type, col_name, value_str;
            bool has_value = false;
            if (fixed_format) {
                FixedFields f = tokenize_fixed(line);
                type = f.f1;
                col_name = f.f3;
                has_value = !f.f4.empty();
                value_str = f.f4;
            } else {
                if (tokens.size() < 3) {
                    throw std::runtime_error("MpsReader: malformed BOUNDS line: " + line);
                }
                type = tokens[0];
                col_name = tokens[2];
                has_value = tokens.size() >= 4;
                if (has_value) value_str = tokens[3];
            }
            if (col_name.empty()) {
                throw std::runtime_error("MpsReader: malformed BOUNDS line: " + line);
            }

            auto cit = col_index.find(col_name);
            if (cit == col_index.end()) {
                throw std::runtime_error("MpsReader: BOUNDS references undeclared column: " +
                                          col_name);
            }
            const auto c = static_cast<std::size_t>(cit->second);
            const double value = has_value ? parse_double(value_str, "BOUNDS value") : 0.0;

            if (type == "UP") {
                model.col_upper[c] = value;
                if (value < 0.0 && !has_explicit_lower[c]) {
                    // Established convention: an UP-only negative bound
                    // with no explicit LO would otherwise leave the
                    // default lower bound (0) above the upper bound.
                    model.col_lower[c] = -kInf;
                }
            } else if (type == "LO") {
                model.col_lower[c] = value;
                has_explicit_lower[c] = true;
            } else if (type == "FX") {
                model.col_lower[c] = value;
                model.col_upper[c] = value;
                has_explicit_lower[c] = true;
            } else if (type == "FR") {
                model.col_lower[c] = -kInf;
                model.col_upper[c] = kInf;
                has_explicit_lower[c] = true;
            } else if (type == "MI") {
                model.col_lower[c] = -kInf;
                has_explicit_lower[c] = true;
            } else if (type == "PL") {
                model.col_upper[c] = kInf;
            } else if (type == "LI") {
                model.col_lower[c] = value;
                model.col_types[c] = VariableType::INTEGER;
                has_explicit_lower[c] = true;
            } else if (type == "UI") {
                model.col_upper[c] = value;
                model.col_types[c] = VariableType::INTEGER;
            } else if (type == "BV") {
                model.col_lower[c] = 0.0;
                model.col_upper[c] = 1.0;
                model.col_types[c] = VariableType::BINARY;
                has_explicit_lower[c] = true;
            } else {
                throw std::runtime_error("MpsReader: unknown BOUNDS type: " + type);
            }
        }
        // ENDATA carries no data.
    }

    if (model.rhs.empty() && model.n_rows > 0) {
        model.rhs.assign(static_cast<std::size_t>(model.n_rows), 0.0);
    }

    if (integer_section) {
        throw std::runtime_error("MpsReader: unterminated INTORG/INTEND marker section");
    }

    return model;
}

MpsModel read_mps_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("MpsReader: cannot open file: " + path);
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) {
        lines.push_back(line);
    }

    try {
        return parse_mps_lines(lines, /*fixed_format=*/false);
    } catch (const std::exception&) {
        // A handful of classic Netlib files need strict fixed-column MPS
        // parsing (see parse_mps_lines' comment); retry once before giving
        // up so a genuinely malformed file still surfaces its real error.
        return parse_mps_lines(lines, /*fixed_format=*/true);
    }
}

} // namespace sihps
