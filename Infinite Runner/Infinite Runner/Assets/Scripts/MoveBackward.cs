using UnityEngine;

public class MoveBackward : MonoBehaviour
{
    [SerializeField]
    private float moveSpeed = 10f;

    private GameObject player;
    private PlayerController playerController;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        player = GameObject.FindWithTag("Player");

        playerController = player.GetComponent<PlayerController>();
    }

    // Update is called once per frame
    void Update()
    {
        if (!playerController.gameOver && playerController.gameStarted)
        {
            transform.Translate(
            Vector3.right * moveSpeed * Time.deltaTime,
            Space.World
        );
        }
    }
}
