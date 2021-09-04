-- 19. Add bonus
DELIMITER $$
CREATE PROCEDURE AddBonus (
    IN user_id INT, project_name CHAR(255), score INT
)
BEGIN
    SET @count = (
        SELECT COUNT(*)
        FROM projects
        WHERE project_name = projects.name
    );
    IF @count = 0 THEN
        INSERT INTO projects (name) VALUES (project_name);
    END IF;
    SET @project = (
        SELECT id
        FROM projects
        WHERE project_name = projects.name
    );
    INSERT INTO corrections VALUES (user_id, @project, score);
END$$
DELIMITER ;