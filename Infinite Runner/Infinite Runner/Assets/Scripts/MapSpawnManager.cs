using System.Collections.Generic;
using UnityEngine;

public class MapSpawnManager : MonoBehaviour
{
    public GameObject[] mapPieces;
    public GameObject player;
    public ObstacleSpawner obstacleSpawner;

    public float startOffset = 5f;

    private Transform currentEndPoint;

    private Queue<GameObject> spawnedPieces = new Queue<GameObject>();

    void Start()
    {
        SpawnPiece();
        SpawnPiece();
        SpawnPiece();
    }

    void Update()
    {
        if (spawnedPieces.Count == 0)
            return;

        GameObject firstPiece =
            spawnedPieces.Peek();

        Transform endPoint =
            firstPiece.transform.Find("endPoint");

        if (endPoint.position.x >
           player.transform.position.x)
        {
            SpawnNewPiece();
        }
    }

    void SpawnNewPiece()
    {
        SpawnPiece();

        GameObject oldPiece =
            spawnedPieces.Dequeue();

        // Optional for later:
        Destroy(oldPiece);
    }

    void SpawnPiece()
    {
        GameObject prefab =
            mapPieces[Random.Range(0, mapPieces.Length)];

        GameObject newPiece =
            Instantiate(prefab);

        Transform startPoint =
            newPiece.transform.Find("startPoint");

        Transform endPoint =
            newPiece.transform.Find("endPoint");

        // First piece
        if (currentEndPoint == null)
        {
            // Spawn in front of player
            Vector3 spawnPos =
                player.transform.position +
                player.transform.right * startOffset - new Vector3(0,0,5.5f);

            // Align StartPoint with spawn position
            Vector3 offset =
                spawnPos - startPoint.position;

            newPiece.transform.position += offset;
        }
        else
        {
            Vector3 offset =
                currentEndPoint.position -
                startPoint.position;

            newPiece.transform.position += offset;
        }

        currentEndPoint = endPoint;

        spawnedPieces.Enqueue(newPiece);

        MapPiece piece = newPiece.GetComponent<MapPiece>();

        obstacleSpawner.SpawnObstacles(piece);
    }

    public void RestartGame()
    {
        while (spawnedPieces.Count > 0)
        {
            GameObject piece =
                spawnedPieces.Dequeue();

            Destroy(piece);
        }
        currentEndPoint = null;

        player.GetComponent<PlayerController>().StartGame();
        SpawnPiece();
        SpawnPiece();
        SpawnPiece();
    }
}
