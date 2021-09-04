-- 6. Temperatures #0
SELECT city, AVG(value) as avg_temp FROM temperatures GROUP BY CITY ORDER BY avg_temp DESC;
