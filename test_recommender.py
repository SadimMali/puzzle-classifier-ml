import unittest
from recommender import cosine_similarity, recommend


def puzzle(id, themes, rating=None):
    return dict(id=id, themes=themes, puzzle_type=None, difficulty_rating=rating)


class RecommendationTests(unittest.TestCase):
    def test_cosine(self):
        self.assertEqual(cosine_similarity({'a': 2}, {'a': 8}), 1)
        self.assertEqual(cosine_similarity({'a': 2}, {'b': 8}), 0)
        self.assertEqual(cosine_similarity({}, {}), 0)

    def test_history_and_duplicates(self):
        a, b = puzzle('a', ['fork']), puzzle('b', ['pin'])
        result = recommend([a, b], [a], [b, a])
        self.assertEqual(result[0]['id'], 'a')
        self.assertEqual(result, recommend([a, b], [a, a], [b, a]))
        self.assertEqual(recommend([a, b], [b], [a, b])[0]['id'], 'b')

    def test_global_idf(self):
        a, b = puzzle('a', ['fork']), puzzle('b', ['pin'])
        catalog = [a, b] + [puzzle(str(i), ['fork']) for i in range(10)]
        result = recommend(catalog, [puzzle('h', ['fork', 'pin'])], [a, b])
        self.assertEqual(result[0]['id'], 'b')
        self.assertEqual(result[0]['recommendation']['globalPuzzleCount'], 12)

    def test_recent_theme_changes_ranking_despite_large_old_mate_history(self):
        mate = puzzle('mate', ['mate', 'mateIn1', 'oneMove'])
        fork = puzzle('fork', ['fork', 'short'])
        old = [dict(mate, id=f'old-{i}', lastAttemptAt='2026-09-01T10:00:00Z') for i in range(50)]
        catalog = [mate, fork]
        self.assertEqual(recommend(catalog, old, [mate, fork])[0]['id'], 'mate')
        recent = dict(fork, id='recent', lastAttemptAt='2026-09-11T10:00:00Z')
        result = recommend(catalog, old + [recent], [mate, fork])
        self.assertEqual(result[0]['id'], 'fork')
        self.assertEqual(result[0]['recommendation']['historyPuzzleCount'], 51)
        self.assertEqual(result, recommend(catalog, [recent] + old, [mate, fork]))
        self.assertEqual(result, recommend(catalog, old + [recent, recent], [mate, fork]))

    def test_retry_uses_latest_timestamp_and_timezone_order(self):
        mate, fork = puzzle('mate', ['mateIn1']), puzzle('fork', ['fork'])
        old_fork = dict(fork, lastAttemptAt='2026-09-01T10:00:00Z')
        new_fork = dict(fork, lastAttemptAt='2026-09-11T16:00:00+05:45')
        recent_mate = dict(mate, lastAttemptAt='2026-09-11T10:00:00Z')
        history = [new_fork, recent_mate, old_fork]
        result = recommend([mate, fork], history, [mate, fork])
        self.assertEqual(result[0]['id'], 'fork')
        self.assertEqual(result[0]['recommendation']['historyPuzzleCount'], 2)

    def test_cold_start_and_empty(self):
        a, b = puzzle('a', ['fork']), puzzle('b', ['pin'])
        result = recommend([a, b, puzzle('c', ['fork'])], [], [b, a])
        self.assertEqual(result[0]['id'], 'a')
        self.assertEqual(result[0]['recommendation']['profile'], 'global')
        self.assertEqual(recommend([], [], []), [])
        self.assertEqual(recommend([], [], [b, a])[0]['id'], 'a')

    def test_difficulty_and_pagination(self):
        a, b = puzzle('a', ['fork'], 1000), puzzle('b', ['fork'], 2000)
        self.assertEqual(recommend([a, b], [b], [a, b])[0]['id'], 'b')
        self.assertEqual(recommend([a, b], [b], [a, b], 1, 1)[0]['id'], 'a')
        self.assertEqual(recommend([a, b], [b], [a, b], 2, 1), [])
        for rating in [None, float('nan'), float('inf'), 1000]:
            p = puzzle('p', [], rating)
            self.assertEqual(recommend([p], [p], [p])[0]['recommendation']['score'], 0)


if __name__ == '__main__':
    unittest.main()
