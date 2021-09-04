-- 11. Rotten tomatoes
SELECT tvs.title, SUM(rate) AS rating FROM tv_shows tvs, tv_show_ratings tvsr WHERE tvs.id=tvsr.show_id GROUP BY tvs.id ORDER BY SUM(rate) DESC;
