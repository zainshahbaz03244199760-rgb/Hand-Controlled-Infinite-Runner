using UnityEngine;

public class ObstacleSpawner : MonoBehaviour
{

    public GameObject[] obstaclePrefabs;

    public void SpawnObstacles(MapPiece piece)
    {
        foreach (Transform spawnPoint in piece.obstacleSpawns)
        {
            // 50% chance to use this point
            if (Random.value < 0.3f)
            {
                GameObject obstacle =
                    obstaclePrefabs[
                        Random.Range(
                            0,
                            obstaclePrefabs.Length
                        )
                    ];

                Instantiate(
                    obstacle,
                    spawnPoint.position + new Vector3(0,0.25f,0),
                    obstacle.transform.rotation,
                    piece.transform
                );
            }
        }
    }

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
