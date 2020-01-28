import psycopg2
import numpy as np

#   connection config
from numpy.linalg import norm

conn = psycopg2.connect(host="127.0.0.1",
                        database="cv_db",
                        user="postgres",
                        password="555555",
                        port="5432")
cur = conn.cursor()


def insert_data(frames, tracked_objects):
    assert len(frames) == len(tracked_objects)

    for frame, objects in zip(frames, tracked_objects):
        for i, obj in enumerate(objects):
            left, top, right, bottom = obj.rect
            label = obj.label
            """Extract the id"""
            id = int(label.split(' ')[-1])
            """Predefined line"""
            line_x1 = 256
            line_y1 = 351
            line_x2 = 481
            line_y2 = 353

            p1 = np.array([line_x1, line_y1])
            p2 = np.array([line_x2, line_y2])
            """Rectangle coord"""

            p3 = np.array([right, bottom])
            # execute the insert query
            dist = norm(np.cross(p2 - p1, p1 - p3)) / norm((p2 - p1))
            # people_exit = 0
            # people_enter = 0
            # if abs(dist[0]) < 20 and dist[0] > 0:
            #    people_exit += 1
            # elif abs(dist[0]) < 20 and dist[0] < 0:
            #    people_enter += 1
            entered = 2
            if dist < 30:
                cur.execute(
                    f"SELECT right_coord, bot_coord from track_data where id={id} order by time_stamps desc limit 10")
                rows = cur.fetchall()

                diff_x = 0
                diff_y = 0

                for r in rows:
                    # print(r[0])
                    diff_x = diff_x + (right - r[0])
                    diff_y = diff_y + (bottom - r[1])
                print(f"Diff_X:{diff_x} and Diff_Y:{diff_y} and ID: {id}")
                if diff_y >= 10 and (left+100) >= line_x1 and (right-100) <= line_x2:
                    entered = 1
                    print(f"{id}: People Entered and dist:{dist}")
                elif diff_y <= -5 and (left+100) >= line_x1 and (right-100) <= line_x2:
                    entered = 0
                    print(f"{id}: People Exit and dist: {dist}")

            # print(f"R:{rows[1]}, id:{id}")

            # print(f"ID: {id}, Distance:{dist}")
            # print(f"People exit: {people_exit}, ID:{id}")
            # print(f"People enter: {people_enter}, ID:{id}")

            # print(f"ID: {id}:Dist: {dist}, entered:{entered}")

            cur.execute(f"INSERT INTO track_data (ID, left_coord, top_coord, right_coord, bot_coord, time_stamps, distance, entered) \
                        VALUES ({id}, {left}, {top}, {right}, {bottom},now(), {dist}, {entered} )")

            conn.commit()
            # print("Values Saved!!!")


def delete_id(id):
    cur.execute(f"DELETE FROM track_data WHERE id={id}")

    conn.commit()


def close_db_conn():
    #   close the db connection
    conn.close()
    print("DB connection closed!!!!")
